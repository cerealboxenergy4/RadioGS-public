"""Smoke test for the new tracer first-hit index (hit_idx) output.
Builds a tiny synthetic scene of 3 axis-aligned opaque surfels and shoots known rays.
Run from the RadioGS repo root (needs assets/bsdf_256_256.bin)."""
import torch
from scene.radiogs_gaussian_model import GaussianModel


def make_model(xyz, scale=0.5, opacity=0.993):
    g = GaussianModel(sh_degree=3)
    N = xyz.shape[0]
    dev = "cuda"
    g._xyz = xyz.to(dev)
    g._scaling = torch.log(torch.full((N, 2), scale, device=dev))           # get_scaling = exp
    quat = torch.zeros(N, 4, device=dev); quat[:, 0] = 1.0                   # identity [w,x,y,z]
    g._rotation = quat
    inv_sig = torch.log(torch.tensor(opacity / (1 - opacity)))
    g._opacity = torch.full((N, 1), float(inv_sig), device=dev)             # get_opacity = sigmoid
    g._features_dc = torch.zeros(N, 1, 3, device=dev)
    g._features_rest = torch.zeros(N, 15, 3, device=dev)
    g._base_color = torch.zeros(N, 3, device=dev)
    g._roughness = torch.zeros(N, 1, device=dev)
    g._metallic = torch.zeros(N, 1, device=dev)
    g.active_sh_degree = 0
    return g


def main():
    torch.cuda.init()
    # 3 surfels lying in the z=0 plane (identity rotation -> normal +z), at distinct x/y.
    xyz = torch.tensor([[0., 0., 0.], [5., 0., 0.], [0., 5., 0.]])
    g = make_model(xyz)
    g.build_bvh()

    # rays from above (z=3) pointing straight down: should hit surfels 0,1, and miss (far away).
    rays_o = torch.tensor([[0., 0., 3.], [5., 0., 3.], [0., 5., 3.], [20., 20., 3.]], device="cuda")
    rays_d = torch.tensor([[0., 0., -1.]] * 4, device="cuda")

    hit_idx, alpha = g.trace_hit_idx(rays_o, rays_d)
    print("hit_idx:", hit_idx.tolist())
    print("alpha:  ", [round(float(a), 3) for a in alpha])
    expected = [0, 1, 2, -1]
    ok = hit_idx.tolist() == expected
    print("dtype:", hit_idx.dtype, "| expected:", expected, "| MATCH:", ok)
    assert hit_idx.dtype == torch.int32
    assert ok, f"hit_idx mismatch: got {hit_idx.tolist()} expected {expected}"

    # also exercise the [N,S] batched-ray shape used by the radiosity build
    rays_o2 = rays_o[:3].unsqueeze(1).repeat(1, 4, 1)   # [3,4,3]
    rays_d2 = rays_d[:3].unsqueeze(1).repeat(1, 4, 1)
    hit2, _ = g.trace_hit_idx(rays_o2.reshape(-1, 3), rays_d2.reshape(-1, 3))
    hit2 = hit2.view(3, 4)
    print("batched hit_idx [3,4]:\n", hit2.tolist())
    assert (hit2 == torch.tensor([[0]*4, [1]*4, [2]*4], device="cuda")).all()
    print("ALL HIT_IDX TESTS PASSED")


if __name__ == "__main__":
    main()
