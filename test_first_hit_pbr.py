"""Smoke test for the first_hit_pbr indirect mode (precompute_first_hit_pbr).

Two opaque surfels facing along +z (surfel 0 at z=0, surfel 1 at z=2) plus a far
spectator; hand-set incident cache. Checks:
  1. buffer shape [N,S,3], finite; fresh model buffer is None (flag-off no-op).
  2. miss rays -> exactly 0.
  3. hit ray matches the reference alpha * (diffuse_out[j] + spec_env) computed
     independently in this test (validates gather indexing, normal flip, vis scaling).
  4. linearity in the radiance cache: doubling the hit surfel's cached radiance moves
     the gathered value by exactly alpha * (albedo_j/pi) * mean(c*area*cos) -- isolates
     the "+1 bounce depth" diffuse path from env/LUT terms.

Run from the RadioGS repo root (needs assets/bsdf_256_256.bin):
  conda run -n radiogs python test_first_hit_pbr.py
"""
import numpy as np
import torch
import nvdiffrast.torch as dr

from scene.radiogs_gaussian_model import GaussianModel
from scene.light import EnvLight
from utils.general_utils import safe_normalize


def make_model(xyz, scale=0.5, opacity=0.993):
    # mirror test_hit_idx.make_model + the material/env fields precompute_first_hit_pbr uses
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
    g._base_color = torch.zeros(N, 3, device=dev)                           # sigmoid -> 0.5
    g._roughness = torch.zeros(N, 1, device=dev)
    g._metallic = torch.zeros(N, 1, device=dev)
    g.active_sh_degree = 0
    g.env_map = EnvLight(path=None, device=dev, resolution=[16, 32], max_res=32,
                         init_value=0.5, activation='exp').cuda()
    return g


def main():
    torch.cuda.init()
    f0 = 0.04
    t_min = 0.05
    # surfel 0 at z=0, surfel 1 directly above at z=2 (both normal +z), spectator far away
    xyz = torch.tensor([[0., 0., 0.], [0., 0., 2.], [50., 50., 0.]])
    g = make_model(xyz)
    g.build_bvh()
    assert g._first_hit_pbr_ind is None, "fresh model must have no first_hit_pbr buffer"

    N, S = 3, 2
    up = torch.tensor([0., 0., 1.], device="cuda")
    down = -up
    dirs = torch.zeros(N, S, 3, device="cuda")
    dirs[0, 0] = up      # surfel 0 ray 0: hits surfel 1
    dirs[0, 1] = down    # surfel 0 ray 1: miss
    dirs[1, :] = up      # surfel 1 rays: miss (nothing above)
    dirs[2, :] = up      # spectator rays: miss
    areas = torch.ones(N, S, 1, device="cuda")
    vis = torch.ones(N, S, 1, device="cuda")
    vis[0] = 0.25                                   # alpha = 0.75 on surfel 0's rays
    rad = torch.zeros(N, S, 3, device="cuda")
    c = torch.tensor([0.3, 0.2, 0.1], device="cuda")
    rad[1] = c                                      # hit surfel's cached indirect radiance

    g.incident_directions = dirs
    g.incident_areas = areas
    g.incident_visibility = vis
    g.incident_radiance = rad

    out = g.precompute_first_hit_pbr(light_t_min=t_min, f0=f0)

    # [1] shape / finiteness / buffer identity
    assert out.shape == (N, S, 3), f"shape {tuple(out.shape)} != {(N, S, 3)}"
    assert torch.isfinite(out).all(), "non-finite entries in _first_hit_pbr_ind"
    assert g._first_hit_pbr_ind is out
    print(f"[1] shape/finite OK  {tuple(out.shape)}")

    # [2] miss rays are exactly zero
    assert torch.all(out[0, 1] == 0), f"miss ray (0,1) nonzero: {out[0,1].tolist()}"
    assert torch.all(out[1] == 0) and torch.all(out[2] == 0), "miss surfels nonzero"
    print("[2] miss rays == 0 OK")

    # [3] hit ray (0,0) against an independent reference for hit surfel j=1
    env = g.get_envmap
    with torch.no_grad():
        albedo = g.get_base_color                                    # [N,3]
        inc1 = vis[1] * env(dirs[1], mode='pure_env') + rad[1]       # [S,3]
        cos1 = (up * dirs[1]).sum(-1, keepdim=True).clamp(min=0)     # [S,1]
        diffuse_out1 = (albedo[1] / np.pi) * (inc1 * areas[1] * cos1).mean(dim=-2)
        # spec at hit: d=+z, wi=-z; normal +z flips to -z to face the ray; ndv=1; refl=-z
        n_j = down
        ndv = torch.ones(1, 1, device="cuda")
        refl = safe_normalize((2 * ndv * n_j - down)[None, :]).squeeze(0)
        rough1 = g.get_rough[1]
        env.build_mips()
        fg_uv = torch.cat([ndv, rough1[None, :].squeeze(-1)[..., None]], dim=-1).clamp(0, 1)
        fg = dr.texture(g.FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(),
                        filter_mode="linear", boundary_mode="clamp").reshape(-1, 2)
        spec = env(refl[None, :], roughness=rough1[None, :], mode='specular')[0] \
            * (f0 * fg[0, 0] + fg[0, 1])
        expected = (1 - vis[0, 0]) * (diffuse_out1 + spec)
    err = (out[0, 0] - expected).abs().max().item()
    assert err < 1e-4, f"hit ray mismatch: got {out[0,0].tolist()} expected {expected.tolist()} (err {err:.2e})"
    print(f"[3] hit-ray gather matches reference (max err {err:.2e})")

    # [4] linearity in the hit surfel's cached radiance (+1-depth diffuse path isolated)
    g.incident_radiance = rad.clone()
    g.incident_radiance[1] = 2 * c
    out2 = g.precompute_first_hit_pbr(light_t_min=t_min, f0=f0)
    with torch.no_grad():
        delta_expected = (1 - vis[0, 0]) * (albedo[1] / np.pi) * (c[None, :] * areas[1] * cos1).mean(dim=-2)
    delta = out2[0, 0] - out[0, 0]
    err = (delta - delta_expected).abs().max().item()
    assert err < 1e-4, f"cache-linearity mismatch: delta {delta.tolist()} expected {delta_expected.tolist()}"
    assert torch.all(out2[0, 1] == 0) and torch.all(out2[1] == 0), "miss rays changed"
    print(f"[4] cache linearity OK (max err {err:.2e})")

    print("test_first_hit_pbr: ALL OK")


if __name__ == "__main__":
    main()
