#
# Differentiable diffuse radiosity solve for RadioGS (single-layer surfel project).
#
# Solves the Lambertian radiosity fixed point over a STATIC sparse transport matrix T
# (built once under frozen geometry from the surfel ray hits):
#
#     L = H + diag(coef) @ (T @ L),     coef = albedo / pi
#       => (I - A) L = H,   A = diag(coef) @ T
#
# L : per-surfel diffuse outgoing radiance [N,3] (view-independent, multi-bounce).
# H : shadowed direct-diffuse outgoing radiance [N,3].
# T : geometry-only transport (sparse [N,N]); T_ij = (1/S) sum_{s: hit(i,s)=j} area*cos.
#
# Forward: truncated Neumann / Jacobi iteration (A is contractive because albedo<1 and
# row-sum(A) = albedo * hemisphere-hit-fraction < 1).
# Backward: exact fixed-point adjoint. Because T is constant (frozen geometry) it is never
# differentiated, so the discontinuous visibility never enters the gradient. We only need
# gradients w.r.t. coef (=> albedo) and H (=> albedo, envmap), via a transposed solve with
# Tt = T.t(), which is free since T is materialized.
import torch


def _spmm(sparse, dense):
    # sparse [N,N] (coalesced) @ dense [N,C] -> [N,C]
    return torch.sparse.mm(sparse, dense)


class RadiosityModule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, T, Tt, coef, H, iters, L_init=None):
        # All math here runs without autograd tracking (custom Function).
        L = H.clone() if L_init is None else L_init.clone()
        for _ in range(iters):
            L = H + coef * _spmm(T, L)
        ctx.save_for_backward(coef, L)
        ctx.T = T
        ctx.Tt = Tt
        ctx.iters = iters
        return L

    @staticmethod
    def backward(ctx, grad_L):
        coef, L = ctx.saved_tensors
        T, Tt, iters = ctx.T, ctx.Tt, ctx.iters
        # Adjoint fixed point: lam = grad_L + Tt @ (coef * lam)  (same contraction, transposed)
        lam = grad_L.clone()
        for _ in range(iters):
            lam = grad_L + _spmm(Tt, coef * lam)
        grad_H = lam                       # dLoss/dH = (I - A)^{-T} grad_L
        grad_coef = lam * _spmm(T, L)      # dLoss/dcoef_i = lam_i * (T L)_i  (per channel)
        # inputs: (T, Tt, coef, H, iters, L_init)
        return None, None, grad_coef, grad_H, None, None
