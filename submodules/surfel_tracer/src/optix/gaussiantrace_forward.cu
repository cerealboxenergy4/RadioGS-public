#include "auxiliary.h"
#include <optix.h>

#include "gaussiantrace_forward.h"

namespace surfel_tracer {

extern "C" {
	__constant__ Gaussiantrace_forward::Params params;
}

extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();

	glm::vec3 ray_o = params.ray_origins[idx.x];
	glm::vec3 ray_d = params.ray_directions[idx.x];
	glm::vec3 ray_origin;

	glm::vec3 C = glm::vec3(0.0f, 0.0f, 0.0f), N = glm::vec3(0.0f, 0.0f, 0.0f);
	float D = 0.0f, O = 0.0f, T = 1.0f, t_start = 0.0f, t_curr = 0.0f;
	float M2 = 0.0f;  // single-layer: Sum_i w_i^2, the per-ray k_eff surrogate denominator
	float F[MAX_FEATURE_SIZE] = {0.0f};

	// single-layer / first-hit instrumentation: per-ray accepted hits (k_eff) and candidate intersections.
	unsigned int khit = 0, kcand = 0;

	// radiosity: gs_idx of the first accepted surfel along this ray (-1 if the ray hits nothing).
	int hit0 = -1;

	HitInfo hitArray[MAX_BUFFER_SIZE];
	unsigned int hitArrayPtr0 = (unsigned int)((uintptr_t)(&hitArray) & 0xFFFFFFFF);
    unsigned int hitArrayPtr1 = (unsigned int)(((uintptr_t)(&hitArray) >> 32) & 0xFFFFFFFF);

	while ((t_start < T_SCENE_MAX) && (T > params.transmittance_min)){
		ray_origin = ray_o + t_start * ray_d;
		
		for (int i = 0; i < MAX_BUFFER_SIZE; ++i) {
			hitArray[i].t = 1e16f;
			hitArray[i].primIdx = -1;
		}
		optixTrace(
			params.handle,
			make_float3(ray_origin.x, ray_origin.y, ray_origin.z),
			make_float3(ray_d.x, ray_d.y, ray_d.z),
			FLT_EPSILON,                // Min intersection distance
			T_SCENE_MAX,               // Max intersection distance
			0.0f,                // rayTime -- used for motion blur
			OptixVisibilityMask(255), // Specify always visible
			OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
			0,                   // SBT offset
			1,                   // SBT stride
			0,                   // missSBTIndex
			hitArrayPtr0,
			hitArrayPtr1
		);

		for (int i = 0; i < MAX_BUFFER_SIZE; ++i) {
			int primIdx = hitArray[i].primIdx;

			if (primIdx == -1) {
				t_curr = T_SCENE_MAX;
				break;
			}
			else{
				t_curr = hitArray[i].t;
				kcand += 1;  // single-layer: a candidate surfel intersection examined on this ray
				int gs_idx = params.gs_idxs[primIdx];

				float o = params.opacity[gs_idx];
				glm::vec3 mean3D = params.means3D[gs_idx];
				glm::vec3 n = params.normals[gs_idx];
				glm::vec3 ru = params.ru[gs_idx];
				glm::vec3 rv = params.rv[gs_idx];

				float cos = -dot(ray_d, n);
				float multiplier = ((cos > 0)) ? 1: -1;
				if ((multiplier < 0) && params.back_culling) continue;
				// glm::vec3 n_flip = multiplier * n;
				glm::vec3 n_flip = n;

				// glm::vec3 n_flip;
				// if (params.back_culling){
				// 	n_flip = n;
				// } 
				// else {
				// 	n_flip = multiplier * n;
				// }
				

				float o_g = glm::dot(n, (ray_o - mean3D));
				float d_g = glm::dot(n, ray_d);
				float d = -o_g * d_g / max(1e-6f, d_g * d_g);
				glm::vec3 pos = ray_o + d * ray_d - mean3D;
				glm::vec2 p_g = {glm::dot(ru, pos), glm::dot(rv, pos)};
				float alpha = min(0.99, o * __expf(super_gaussian_power(glm::dot(p_g, p_g), params.super_gaussian_order)));

				if (alpha<params.alpha_min) continue;

				// radiosity: record the first surfel that passes back-culling + alpha acceptance.
				if (hit0 < 0) hit0 = gs_idx;

				glm::vec3 c = computeColorFromSH_forward(params.deg, ray_d, params.shs + gs_idx * params.max_coeffs);
				if (params.back_culling){
					c = (cos > 0) ? c : glm::vec3(0.0f, 0.0f, 0.0f);
				}

				float w = T * alpha;
				C += w * c;
				N += w * n_flip;
				D += w * d;
				O += w;
				M2 += w * w;  // single-layer: accumulate Sum w_i^2 for the k_eff surrogate

				for (int j = 0; j < params.S; ++j){
					F[j] += w * params.features[gs_idx * params.S + j];
				}

				T *= (1 - alpha);
				khit += 1;  // single-layer: an accepted (contributing) surfel on this ray -> k_eff

				// First-hit north-star: terminate the ray at its first accepted surfel (k_eff := 1).
				// When the scene is ironed so one near-opaque surfel sits at each intersection, this
				// loses almost no energy while collapsing the per-ray sort/accumulate cost.
				if (params.first_hit_only){
					T = 0.0f;
				}

				if (T < params.transmittance_min){
					break;
				}
			}
			

		}
		t_start += t_curr;
	}
	
	params.color[idx.x] = C;
	params.normal[idx.x] = N;
	params.depth[idx.x] = D;
	params.alpha[idx.x] = O;
	params.alpha_m2[idx.x] = M2;  // single-layer: k_eff surrogate second moment
	if (params.hit_idx != nullptr){
		params.hit_idx[idx.x] = hit0;  // radiosity: first-accepted surfel gs_idx (-1 if miss)
	}
	for (int i = 0; i < params.S; ++i){
		params.feature[idx.x * params.S + i] = F[i];
	}

	// single-layer: accumulate global trace-cost counters (candidates, accepted hits = k_eff).
	// Optional: only when a counter buffer was provided (eval-time measurement, off in training).
	if (params.counters != nullptr){
		atomicAdd(params.counters + 0, (unsigned long long)kcand);
		atomicAdd(params.counters + 1, (unsigned long long)khit);
		atomicAdd(params.counters + 2, (unsigned long long)1);  // ray count (rays that ran)
	}
}

extern "C" __global__ void __miss__ms() {
}

extern "C" __global__ void __closesthit__ch() {
}

extern "C" __global__ void __anyhit__ah() {
    HitInfo* hitArray = (HitInfo*)((uintptr_t)optixGetPayload_0() | ((uintptr_t)optixGetPayload_1() << 32));

	float THit = optixGetRayTmax();
    int i_prim = optixGetPrimitiveIndex();
	HitInfo newHit = {THit, i_prim};

	for (int i = 0; i < MAX_BUFFER_SIZE; ++i) {
		if (hitArray[i].primIdx == -1){
			hitArray[i] = newHit;
			break;
		}
        else if (hitArray[i].t > newHit.t) {
			host_device_swap<HitInfo>(hitArray[i], newHit);
        }
    }

	if (THit < hitArray[MAX_BUFFER_SIZE - 1].t) {
        optixIgnoreIntersection(); 
    }

}

}
