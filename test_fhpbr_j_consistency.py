"""Smoke test for the fhpbr j-consistency machinery (first_hit_pbr_outgoing).

Same synthetic scene as test_first_hit_pbr (two opaque surfels + far spectator,
hand-set incident cache). Checks:
  1. equivalence: first_hit_pbr_outgoing(j, wo) * (1 - vis) reproduces the validated
     precompute_first_hit_pbr buffer row for the hit ray -- the helper and the gather
     share formulas by construction; this catches drift between the two code paths.
  2. dedup correctness: repeated j entries return identical rows (unique/inverse path).
  3. gradients: the RHS is differentiable w.r.t. the hit surfel's material (albedo via
     the diffuse path, roughness via the split-sum spec path); detach_mat kills both.

Run from the RadioGS repo root (needs assets/bsdf_256_256.bin):
  conda run -n radiogs python test_fhpbr_j_consistency.py
"""
import torch

from test_first_hit_pbr import make_model


def main():
    torch.cuda.init()
    f0 = 0.04
    t_min = 0.05
    xyz = torch.tensor([[0., 0., 0.], [0., 0., 2.], [50., 50., 0.]])
    g = make_model(xyz)
    g.build_bvh()

    N, S = 3, 2
    up = torch.tensor([0., 0., 1.], device="cuda")
    down = -up
    dirs = torch.zeros(N, S, 3, device="cuda")
    dirs[0, 0] = up      # surfel 0 ray 0: hits surfel 1
    dirs[0, 1] = down    # miss
    dirs[1, :] = up      # miss
    dirs[2, :] = up      # miss
    areas = torch.ones(N, S, 1, device="cuda")
    vis = torch.ones(N, S, 1, device="cuda")
    vis[0] = 0.25
    rad = torch.zeros(N, S, 3, device="cuda")
    rad[1] = torch.tensor([0.3, 0.2, 0.1], device="cuda")

    g.incident_directions = dirs
    g.incident_areas = areas
    g.incident_visibility = vis
    g.incident_radiance = rad

    # reference: the validated gather (traces to find j=1 for ray (0,0), bakes the buffer)
    buf = g.precompute_first_hit_pbr(light_t_min=t_min, f0=f0)

    # [1] helper == gather on the hit ray: j=1, wo = -d = down, scale by (1 - vis)
    g.get_envmap.build_mips()
    j = torch.tensor([1], device="cuda", dtype=torch.long)
    wo = down[None, :]
    with torch.no_grad():
        L = g.first_hit_pbr_outgoing(j, wo, f0=f0)
    got = (L[0] * (1 - vis[0, 0]))
    err = (got - buf[0, 0]).abs().max().item()
    assert err < 1e-5, f"helper != gather: {got.tolist()} vs {buf[0,0].tolist()} (err {err:.2e})"
    print(f"[1] helper matches precompute_first_hit_pbr buffer row (max err {err:.2e})")

    # [2] dedup: repeated j -> identical rows, equal to the single-entry result
    with torch.no_grad():
        L2 = g.first_hit_pbr_outgoing(torch.tensor([1, 1], device="cuda"), wo.repeat(2, 1), f0=f0)
    assert torch.allclose(L2[0], L2[1]), "repeated j rows differ"
    assert torch.allclose(L2[0], L[0]), "dedup path changed the value"
    print("[2] unique/inverse dedup OK")

    # [3] gradient flow to the hit surfel's material; detach_mat kills it
    g._base_color.requires_grad_(True)
    g._roughness.requires_grad_(True)
    out = g.first_hit_pbr_outgoing(j, wo, f0=f0)
    out.sum().backward()
    bc_grad = g._base_color.grad
    r_grad = g._roughness.grad
    assert bc_grad is not None and bc_grad[1].abs().sum() > 0, "no albedo grad at hit surfel"
    assert r_grad is not None and r_grad[1].abs().sum() > 0, "no roughness grad at hit surfel"
    assert bc_grad[0].abs().sum() == 0 and bc_grad[2].abs().sum() == 0, "grad leaked to non-hit surfels"
    g._base_color.grad = None
    g._roughness.grad = None
    g.get_envmap.build_mips()   # fresh mip graph (the first backward freed the shared one)
    out_d = g.first_hit_pbr_outgoing(j, wo, f0=f0, detach_mat=True)
    # env path still differentiable, but material grads must be gone
    if out_d.requires_grad:
        out_d.sum().backward()
    assert g._base_color.grad is None or g._base_color.grad.abs().sum() == 0, "detach_mat leaked albedo grad"
    assert g._roughness.grad is None or g._roughness.grad.abs().sum() == 0, "detach_mat leaked roughness grad"
    print("[3] material gradients flow to the hit surfel only; detach_mat kills them")

    print("test_fhpbr_j_consistency: ALL OK")


if __name__ == "__main__":
    main()
