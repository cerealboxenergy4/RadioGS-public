#pragma once
#include "auxiliary.h"

#include <optix.h>

namespace surfel_tracer {

struct Gaussiantrace_forward {
	struct Params {
		const glm::vec3* ray_origins;
		const glm::vec3* ray_directions;
		const int* gs_idxs;
		const glm::vec3* means3D;
		const float* opacity;
		const glm::vec3* ru;
		const glm::vec3* rv;
		const glm::vec3* normals;
		const float* features;
		const glm::vec3* shs;
		glm::vec3* color;
		glm::vec3* normal;
		float* feature;
		float* depth;
		float* alpha;
		float* alpha_m2;                  // single-layer: per-ray second moment Sum_i w_i^2 (k_eff surrogate)
		int* hit_idx;                     // radiosity: per-ray first-accepted surfel gs_idx (-1 if none), nullable
		int* prefix_idx;                  // virtual-surfel: first n_prefix accepted gs_idx per ray (row-major [ray,K], -1 fill), nullable
		float* prefix_w;                  // virtual-surfel: matching composite weights w=T*alpha at acceptance, nullable
		int n_prefix;                     // virtual-surfel: prefix capacity K (0 = disabled)
		float alpha_min;
		float transmittance_min;
		int deg;
		int max_coeffs;
		int S;
		bool back_culling;
		float super_gaussian_order;
		bool first_hit_only;              // single-layer: terminate each ray at its first accepted surfel
		int hit_buffer_size;              // trace-opt: active K-nearest gather size per traversal round
		                                  // (1..MAX_BUFFER_SIZE). Smaller K commits tmax sooner on ironed
		                                  // scenes (low k_eff) so the BVH culls the tail; exact for any K
		                                  // because the outer relaunch loop resumes past the K-th hit.
		unsigned long long* counters;     // single-layer: [candidates, accepted_hits(k_eff), rays] (nullable)
		OptixTraversableHandle handle;
	};

	struct RayGenData {};
	struct MissData {};
	struct HitGroupData {};
};

}
