"""End-to-end integration check on a real TensoIR hotdog model:
render relight WITHOUT vs WITH the radiosity solve and verify
  - both images are finite,
  - the DIRECT diffuse component is invariant (no double-count),
  - the indirect / full image actually changes (injection is live).
Run from RadioGS repo root in the radiogs env."""
import os, json
import numpy as np
import torch
from argparse import ArgumentParser
from arguments import PipelineParams
from scene.radiogs_gaussian_model import GaussianModel
from scene.light import EnvLight
from scene.cameras import Camera
from gaussian_renderer import render_radiogs
from gaussian_renderer.radiogs import sample_incident_rays
from utils.graphics_utils import focal2fov, fov2focal

MODEL = "outputs/part2/radiogs_tir_hotdog_stock/radiogs"
PLY = os.path.join(MODEL, "point_cloud", "iteration_20000", "point_cloud.ply")
ENV = "data/Environment_Maps/high_res_envmaps_2k/bridge.hdr"
TRANSFORMS = "data/TensoIR/hotdog/transforms_test.json"


def default_pipe():
    p = ArgumentParser()
    pp = PipelineParams(p)
    return pp.extract(p.parse_args([]))


def main():
    pipe = default_pipe()
    if not hasattr(pipe, "depth_ratio"):
        pipe.depth_ratio = 0.0
    pipe.diffuse_sample_num = 128
    pipe.light_sample_num = 0
    pipe.radiosity_solver_iters = 16

    g = GaussianModel(3)
    g.super_gaussian_order = getattr(pipe, "super_gaussian_order", 2.0)
    g.first_hit_only = getattr(pipe, "first_hit_only", False)
    g.load_ply(PLY)
    g.active_sh_degree = g.max_sh_degree
    g.build_bvh()
    g.env_map = EnvLight(path=ENV, device="cuda", max_res=1024, activation="none").cuda()
    g.env_map.build_mips(); g.env_map.update_pdf()
    g.env_map.set_transform(torch.tensor([[0,-1,0],[0,0,1],[-1,0,0]], dtype=torch.float32, device="cuda"))
    print(f"loaded N={g.get_xyz.shape[0]} surfels")

    # camera from test frame 0
    frame = json.load(open(TRANSFORMS))
    fovx = frame["camera_angle_x"]; f0 = frame["frames"][0]
    c2w = np.array(f0["transform_matrix"]); c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w); R = np.transpose(w2c[:3, :3]); T = w2c[:3, 3]
    H = W = 800
    fovy = focal2fov(fov2focal(fovx, W), H)
    cam = Camera(colmap_id=0, R=R, T=T, FoVx=fovx, FoVy=fovy,
                 image=torch.zeros(3, H, W), gt_alpha_mask=None, image_name=None, uid=0)

    bg = torch.zeros(3, device="cuda")
    bcs = torch.ones(3, device="cuda")

    # incident cache (diffuse rays), relight one-bounce radiance for specular
    with torch.no_grad():
        dir_pp = g.get_xyz - cam.camera_center
        dir_pp = dir_pp / (dir_pp.norm(dim=-1, keepdim=True) + 1e-6)
        normal_dummy = g.get_normal(scaling_modifier=1.0, dir_pp_normalized=dir_pp)
        dirs, areas = sample_incident_rays(normal_dummy, False, pipe.diffuse_sample_num)
        g.update_incidents_directions(dirs, areas)
        feats = torch.cat([g.get_base_color * bcs, g.get_rough], dim=1)
        g.precompute_incidents(light_t_min=pipe.light_t_min, only_vis=False, features=feats,
                               relight=True, back_culling=pipe.back_culling)

    rk = dict(pc=g, pipe=pipe, bg_color=bg, training=False, relight=True, base_color_scale=bcs)

    # --- flag OFF (baseline one-bounce) ---
    pipe.use_radiosity_solve = False
    g._radiosity_indirect = None
    with torch.no_grad():
        out0 = render_radiogs(viewpoint_camera=cam, **rk)
    # --- build T + solve, flag ON ---
    pipe.use_radiosity_solve = True
    with torch.no_grad():
        g.build_radiosity_transport(light_t_min=pipe.light_t_min, back_culling=pipe.back_culling)
        ind = g.solve_diffuse_radiosity(iters=pipe.radiosity_solver_iters, base_color_scale=bcs, differentiable=False)
        out1 = render_radiogs(viewpoint_camera=cam, **rk)

    def stat(t): return f"finite={torch.isfinite(t).all().item()} min={t.min():.4f} max={t.max():.4f} mean={t.mean():.4f}"
    print("indirect (per-surfel):", stat(ind), "nnz_T=", g.radiosity_T._nnz())
    print("render OFF:", stat(out0["render"]))
    print("render ON :", stat(out1["render"]))
    d0, d1 = out0["render_direct"], out1["render_direct"]
    direct_diff = (d0 - d1).abs().max().item()
    full_diff = (out0["render"] - out1["render"]).abs().max().item()
    ind_diff = (out0["render_indirect"] - out1["render_indirect"]).abs().max().item()
    print(f"max|render_direct OFF-ON|   = {direct_diff:.3e}  (expect ~0: direct invariant)")
    print(f"max|render_indirect OFF-ON| = {ind_diff:.3e}  (expect >0: indirect changed)")
    print(f"max|render OFF-ON|          = {full_diff:.3e}  (expect >0)")
    assert torch.isfinite(out1["render"]).all(), "non-finite render with radiosity"
    assert direct_diff < 1e-4, f"direct term changed (double-count?): {direct_diff}"
    assert full_diff > 1e-4, "radiosity injection had no effect"
    print("INTEGRATION TEST PASSED")


if __name__ == "__main__":
    main()
