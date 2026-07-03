import torch
from surfel_tracer import _C


class _GaussianTrace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bvh, rays_o, rays_d, gs_idxs, means3D, opacity, ru, rv, normals, features, shs, alpha_min, transmittance_min, deg, back_culling, super_gaussian_order, first_hit_only, counters, n_prefix):
        color = torch.zeros_like(rays_o)
        normal = torch.zeros_like(rays_o)
        feature = torch.zeros(*rays_o.shape[:-1], features.shape[-1], device=rays_o.device, dtype=rays_o.dtype)
        depth = torch.zeros_like(rays_o[:, 0])
        alpha = torch.zeros_like(rays_o[:, 0])
        alpha_m2 = torch.zeros_like(rays_o[:, 0])  # single-layer: per-ray Sum w_i^2 (k_eff surrogate)
        # radiosity: per-ray first-accepted surfel gs_idx (-1 if the ray hits nothing)
        hit_idx = torch.full((rays_o.shape[0],), -1, dtype=torch.int32, device=rays_o.device)
        # virtual-surfel: per-ray significant-hit prefix (first n_prefix accepted gs_idx + weights);
        # (B,0) when disabled so the kernel receives nullptr. prefix[:,0] == hit_idx (first-accepted).
        K = int(n_prefix) if n_prefix and n_prefix > 0 else 0
        prefix_idx = torch.full((rays_o.shape[0], K), -1, dtype=torch.int32, device=rays_o.device)
        prefix_w = torch.zeros((rays_o.shape[0], K), dtype=rays_o.dtype, device=rays_o.device)
        # single-layer: counters is a 3-long int64 tensor [candidates, accepted_hits, rays] or empty (off)
        if counters is None:
            counters = torch.empty(0, dtype=torch.int64, device=rays_o.device)
        bvh.trace_forward(
            rays_o, rays_d, gs_idxs, means3D, opacity, ru, rv, normals, features, shs,
            color, normal, feature, depth, alpha, alpha_m2, hit_idx, prefix_idx, prefix_w,
            alpha_min, transmittance_min, deg, back_culling, super_gaussian_order,
            first_hit_only, counters, K
        )

        ctx.alpha_min = alpha_min
        ctx.transmittance_min = transmittance_min
        ctx.deg = deg
        ctx.bvh = bvh
        ctx.back_culling = back_culling
        ctx.super_gaussian_order = super_gaussian_order
        ctx.save_for_backward(rays_o, rays_d, gs_idxs, means3D, opacity, ru, rv, normals, features, shs, color, normal, feature, depth, alpha, alpha_m2)
        return color, normal, feature, depth, alpha, alpha_m2, hit_idx, prefix_idx, prefix_w

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_normal, grad_out_feature, grad_out_depth, grad_out_alpha, grad_out_alpha_m2, grad_out_hit_idx, grad_out_prefix_idx, grad_out_prefix_w):
        rays_o, rays_d, gs_idxs, means3D, opacity, ru, rv, normals, features, shs, color, normal, feature, depth, alpha, alpha_m2 = ctx.saved_tensors
        grad_rays_o = torch.zeros_like(rays_o)
        grad_rays_d = torch.zeros_like(rays_d)
        grad_means3D = torch.zeros_like(means3D)
        grad_opacity = torch.zeros_like(opacity)
        grad_ru = torch.zeros_like(ru)
        grad_rv = torch.zeros_like(rv)
        grad_normals = torch.zeros_like(normals)
        grad_features = torch.zeros_like(features)
        grad_shs = torch.zeros_like(shs)
        
        ctx.bvh.trace_backward(
            rays_o, rays_d, gs_idxs, means3D, opacity, ru, rv, normals, features, shs,
            color, normal, feature, depth, alpha, alpha_m2,
            grad_rays_o, grad_rays_d, grad_means3D, grad_opacity, grad_ru, grad_rv, grad_normals, grad_features, grad_shs,
            grad_out_color, grad_out_normal, grad_out_feature, grad_out_depth, grad_out_alpha, grad_out_alpha_m2.contiguous(),
            ctx.alpha_min, ctx.transmittance_min, ctx.deg, ctx.back_culling, ctx.super_gaussian_order
        )
        
        grads = (
            None,
            grad_rays_o,
            grad_rays_d,
            None,
            grad_means3D,
            grad_opacity,
            grad_ru,
            grad_rv,
            grad_normals,
            grad_features,
            grad_shs,
            None,
            None,
            None,
            None,
            None,  # super_gaussian_order
            None,  # first_hit_only
            None,  # counters
            None,  # n_prefix
        )

        return grads


class GaussianTracer():
    def __init__(self, transmittance_min=0.001):
        self.impl = _C.create_gaussiantracer()
        self.transmittance_min = transmittance_min
        # single-layer: trace-cost instrumentation. When collect_counters is True, trace() accumulates
        # [candidate_intersections, accepted_hits (k_eff), rays] into counter_accum across calls.
        self.collect_counters = False
        self.counter_accum = None

    def reset_counters(self):
        self.counter_accum = torch.zeros(3, dtype=torch.int64, device='cuda')

    def read_counters(self):
        """Return (mean_candidates_per_ray, mean_keff_per_ray, n_rays). k_eff = mean accepted hits/ray."""
        if self.counter_accum is None:
            return (0.0, 0.0, 0)
        cand, khit, rays = [int(x) for x in self.counter_accum.tolist()]
        if rays == 0:
            return (0.0, 0.0, 0)
        return (cand / rays, khit / rays, rays)

    def build_bvh(self, vertices_b, faces_b, gs_idxs):
        self.faces_b = faces_b
        self.gs_idxs = gs_idxs.int()
        self.impl.build_bvh(vertices_b[faces_b])

    def update_bvh(self, vertices_b, faces_b, gs_idxs):
        assert (self.faces_b == faces_b).all(), "Update bvh must keep the triangle id not change~"
        self.gs_idxs = gs_idxs.int()
        self.impl.update_bvh(vertices_b[faces_b])

    def trace(self, rays_o, rays_d, means3D, opacity, ru, rv, normals, features, shs, alpha_min, deg=3, back_culling=False, super_gaussian_order=2.0, first_hit_only=False, return_alpha_m2=False, return_hit_idx=False, return_prefix=False, n_prefix=8):
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        means3D = means3D.contiguous()
        opacity = opacity.contiguous()
        ru = ru.contiguous()
        rv = rv.contiguous()
        normals = normals.contiguous()
        if features is not None:
            features = features.contiguous()
        else:
            features = torch.zeros_like(means3D[:, :0])
        shs = shs.contiguous()

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        B = rays_o.shape[0]
        mask = torch.zeros(B, dtype=torch.bool, device='cuda')
        self.impl.intersection_test(rays_o, rays_d, self.gs_idxs, means3D, opacity, ru, rv, normals, mask)
        color = torch.zeros(B, 3, dtype=torch.float32, device='cuda')
        normal = torch.zeros(B, 3, dtype=torch.float32, device='cuda')
        feature = torch.zeros(B, features.shape[-1], dtype=torch.float32, device='cuda')
        depth = torch.zeros(B, dtype=torch.float32, device='cuda')
        alpha = torch.zeros(B, dtype=torch.float32, device='cuda')
        alpha_m2 = torch.zeros(B, dtype=torch.float32, device='cuda')  # single-layer: Sum w_i^2
        hit_idx = torch.full((B,), -1, dtype=torch.int32, device='cuda')  # radiosity: first-hit gs_idx (-1 = miss)
        # virtual-surfel: per-ray significant-hit prefix buffers (K=0 -> disabled -> kernel gets nullptr)
        K = int(n_prefix) if return_prefix and n_prefix and n_prefix > 0 else 0
        prefix_idx = torch.full((B, K), -1, dtype=torch.int32, device='cuda')
        prefix_w = torch.zeros((B, K), dtype=torch.float32, device='cuda')

        rays_o_ = rays_o[mask]
        rays_d_ = rays_d[mask]
        # single-layer: counter buffer for k_eff / candidate instrumentation (None -> disabled)
        counters = None
        if self.collect_counters:
            if self.counter_accum is None:
                self.reset_counters()
            counters = self.counter_accum
        if not rays_o_.shape[0] == 0:
            color[mask], normal[mask], feature[mask], depth[mask], alpha[mask], alpha_m2[mask], hit_idx[mask], prefix_idx[mask], prefix_w[mask] = _GaussianTrace.apply(self.impl, rays_o_, rays_d_, self.gs_idxs, means3D, opacity, ru, rv, normals, features, shs, alpha_min, self.transmittance_min, deg, back_culling, super_gaussian_order, first_hit_only, counters, K)

        color = color.view(*prefix, 3)
        normal = normal.view(*prefix, 3)
        feature = feature.view(*prefix, features.shape[-1])
        depth = depth.view(*prefix)
        alpha = alpha.view(*prefix)
        alpha_m2 = alpha_m2.view(*prefix)
        hit_idx = hit_idx.view(*prefix)
        prefix_idx = prefix_idx.view(*prefix, K)
        prefix_w = prefix_w.view(*prefix, K)

        if return_prefix:
            # significant-hit prefix for the virtual-surfel / prefix-composite fhpbr gathers.
            return color, normal, feature, depth, alpha, prefix_idx, prefix_w
        if return_hit_idx:
            return color, normal, feature, depth, alpha, hit_idx
        if return_alpha_m2:
            return color, normal, feature, depth, alpha, alpha_m2
        return color, normal, feature, depth, alpha