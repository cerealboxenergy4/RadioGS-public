"""Unit tests for the differentiable radiosity solver.
1. Forward: truncated Neumann matches the dense linear solve (I-A)L=H.
2. Backward: exact fixed-point adjoint matches autograd through a dense differentiable solve.
Run from RadioGS repo root in the radiogs env."""
import torch
from utils.radiosity import RadiosityModule


def test_module():
    torch.manual_seed(0)
    dev = "cuda"
    N = 40
    Td = (torch.rand(N, N, device=dev) * 0.02)          # small entries -> contractive
    Td.fill_diagonal_(0.0)
    T = Td.to_sparse().coalesce()
    Tt = T.t().coalesce()
    coef = torch.rand(N, 3, device=dev) * 0.8           # albedo/pi-like, <1
    H = torch.rand(N, 3, device=dev)
    W = torch.rand(N, 3, device=dev)                    # arbitrary loss weights
    iters = 800
    I = torch.eye(N, device=dev)

    # ---- forward vs dense solve ----
    L_mod = RadiosityModule.apply(T, Tt, coef, H, iters, None)
    L_ref = torch.zeros(N, 3, device=dev)
    for c in range(3):
        A = torch.diag(coef[:, c]) @ Td
        L_ref[:, c] = torch.linalg.solve(I - A, H[:, c])
    fwd_err = (L_mod - L_ref).abs().max().item()
    print(f"[forward] max|L_mod - L_ref| = {fwd_err:.3e}")
    assert fwd_err < 1e-4, fwd_err

    # ---- backward (module, exact adjoint) ----
    coef_m = coef.clone().requires_grad_(True)
    H_m = H.clone().requires_grad_(True)
    L_m = RadiosityModule.apply(T, Tt, coef_m, H_m, iters, None)
    (W * L_m).sum().backward()

    # ---- backward (reference, autograd through dense linalg.solve) ----
    coef_r = coef.clone().requires_grad_(True)
    H_r = H.clone().requires_grad_(True)
    cols = []
    for c in range(3):
        A = torch.diag(coef_r[:, c]) @ Td
        cols.append(torch.linalg.solve(I - A, H_r[:, c]))
    L_r = torch.stack(cols, dim=1)
    (W * L_r).sum().backward()

    gH_err = (H_m.grad - H_r.grad).abs().max().item()
    gC_err = (coef_m.grad - coef_r.grad).abs().max().item()
    print(f"[backward] max|grad_H  diff| = {gH_err:.3e}")
    print(f"[backward] max|grad_coef diff| = {gC_err:.3e}")
    assert gH_err < 1e-4, gH_err
    assert gC_err < 1e-4, gC_err
    print("RADIOSITY MODULE FORWARD + ADJOINT TESTS PASSED")


def test_build_and_solve():
    """Smoke: build T on a tiny scene and solve; check shapes/finiteness/non-negativity/contraction."""
    from scene.radiogs_gaussian_model import GaussianModel
    from scene.light import EnvLight
    import numpy as np

    dev = "cuda"
    # floor surfel (0) + two surfels above it (1,2). Floor faces up; uppers face down toward floor.
    g = GaussianModel(sh_degree=3)
    N = 3
    g._xyz = torch.tensor([[0., 0., 0.], [0.3, 0., 1.0], [-0.3, 0., 1.0]], device=dev)
    g._scaling = torch.log(torch.full((N, 2), 1.0, device=dev))
    quat = torch.zeros(N, 4, device=dev); quat[:, 0] = 1.0      # identity -> normal +z
    # flip the two upper surfels to face down (-z): rotate 180 about x -> quat (0,1,0,0)
    quat[1] = torch.tensor([0., 1., 0., 0.], device=dev)
    quat[2] = torch.tensor([0., 1., 0., 0.], device=dev)
    g._rotation = quat
    inv_sig = float(torch.log(torch.tensor(0.99 / 0.01)))
    g._opacity = torch.full((N, 1), inv_sig, device=dev)
    g._features_dc = torch.zeros(N, 1, 3, device=dev)
    g._features_rest = torch.zeros(N, 15, 3, device=dev)
    g._base_color = torch.zeros(N, 3, device=dev)              # pre-activation -> albedo ~0.5
    g._roughness = torch.zeros(N, 1, device=dev)
    g._metallic = torch.zeros(N, 1, device=dev)
    g.active_sh_degree = 0
    g.build_bvh()

    # sample a hemisphere of incident directions per surfel (use the model's own sampler)
    from gaussian_renderer.radiogs import sample_incident_rays
    splat2world = g.get_covariance()
    from utils.general_utils import safe_normalize
    normals = safe_normalize(splat2world[:, 2, :3])
    dirs, areas = sample_incident_rays(normals, is_training=False, sample_num=64)
    g.update_incidents_directions(dirs, areas)
    g.precompute_incidents(light_t_min=0.05, only_vis=True)

    T = g.build_radiosity_transport(light_t_min=0.05)
    print(f"[build] T shape={tuple(T.shape)} nnz={T._nnz()}")
    rowsum = torch.sparse.sum(T, dim=1).to_dense()
    print(f"[build] T row-sum: min={rowsum.min():.3f} max={rowsum.max():.3f} (<= ~pi)")
    assert T.shape == (N, N)
    assert torch.isfinite(rowsum).all() and (rowsum >= 0).all()

    # need an envmap for the direct term H; use a constant-ish env via a real hdr if available
    env_path = "data/Environment_Maps/high_res_envmaps_2k/bridge.hdr"
    import os
    if os.path.exists(env_path):
        g.env_map = EnvLight(path=env_path, device='cuda', max_res=512, activation='none').cuda()
        g.env_map.build_mips(); g.env_map.update_pdf()
        ind = g.solve_diffuse_radiosity(iters=24, differentiable=False)
        print(f"[solve] indirect shape={tuple(ind.shape)} min={ind.min():.4f} max={ind.max():.4f}")
        assert ind.shape == (N, 3)
        assert torch.isfinite(ind).all() and (ind >= 0).all()
        print("BUILD + SOLVE SMOKE PASSED")
    else:
        print("(skipping solve: no envmap found)")


if __name__ == "__main__":
    test_module()
    test_build_and_solve()
