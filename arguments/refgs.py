#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, nargs="+")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, default=value, nargs="+")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        # Rendering Settings
        self.sh_degree = 3
        self._resolution = -1
        self._white_background = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        self.batch_size = 2**16
        
        # Paths
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self.init_ply = ""   # if set, overrides dataset points3d.ply with this EDGS/custom PLY

        # Device Settings
        self.data_device = "cuda"
        self.eval = False

        # EnvLight Settings
        self.envmap_resolution = 128
        self.envmap_max_roughness = 0.5
        self.envmap_min_roughness = 0.08
        self.relight = False
        self.envmap_activation = 'sigmoid'  # 'sigmoid' or 'none'

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path)
        return group


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        # Processing Settings
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.use_asg = False
        self.bf_random = False

        # Debugging
        self.depth_ratio = 0.0
        self.debug = False
        # single-layer: super-Gaussian footprint order for the rasterizer (2.0 == standard Gaussian)
        self.super_gaussian_order = 2.0
        # Resume model parameters with a fresh optimizer so continuation runs
        # honor the learning rates passed on the new command line.
        self.restart = False

        # SRGB Transformation
        self.srgb = False

        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # Learning Rate Settings
        self.iterations = 50_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.features_lr = 0.0075 
        self.indirect_lr = 0.0075 
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001

        self.base_color_lr = 0.0075 
        self.metallic_lr =  0.0
        self.roughness_lr =  0.005 
        self.normal_lr = 0.006
        self.envmap_cubemap_lr = 0.01
        
        # Densification Settings
        self.percent_dense = 0.01

        # Regularization Parameters
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal_render_depth = 0.05
        self.lambda_normal_smooth = 0.0
        self.lambda_depth_smooth = 0.0
        self.lambda_mask_entropy = 0.0

        # ---- single-layer ironing (ported from single-layer-surfel Part 1) ----
        self.lambda_single = 0.0            # 0 = ironing off; ironed point e.g. 0.004
        self.single_warmup_iters = 15_000   # off until here (let geometry/normals settle first)
        self.single_ramp_iters = 3_000
        self.single_until_iter = 0          # 0 = hold to end (no decay)
        self.single_decay_iters = 0
        self.single_alpha_thresh = 0.5
        # Confidence-aware selective ironing.  The confidence is detached and only redistributes
        # the N_eff pressure over foreground pixels; its weighted mean keeps lambda_single's scale.
        #   none     : original uniform foreground penalty (bit-exact legacy path)
        #   edge     : protect GT image edges and uncertain silhouette pixels
        #   geometry : protect depth-mixture / rendered-vs-depth-normal disagreement
        #   hybrid   : geometric mean of edge + geometry confidence (recommended)
        self.single_confidence_mode = "none"
        self.single_confidence_edge_tau = 0.05
        self.single_confidence_dist_scale = 1.0
        self.single_confidence_normal_tau = 0.15
        self.single_confidence_floor = 0.05
        # Adaptive capacity routing.  `counterfactual` permits K*(x) in [1, 2] only when full
        # compositing beats the first-hit render and the expected depth is geometrically separated
        # from the first surface.  All routing evidence is detached.
        self.single_layer_target_mode = "fixed"  # fixed | counterfactual
        self.single_counterfactual_benefit_tau = 0.01
        self.single_counterfactual_benefit_temperature = 0.005
        self.single_counterfactual_depth_tau = 0.02
        self.single_counterfactual_depth_temperature = 0.01
        self.single_counterfactual_max_layers = 2.0
        self.single_counterfactual_gate_mode = "soft"  # soft | hard

        # Co-contributor normal alignment for the Stage-1 causal extension.
        # D_n = 1 - ||sum_i w_i n_i / sum_i w_i||^2 is the weighted
        # directional variance of normals intersected by a camera ray.
        self.lambda_response_align_normal = 0.0
        self.response_align_normal_warmup_iters = 0
        self.response_align_normal_ramp_iters = 0
        self.response_align_normal_until_iter = 0
        self.response_align_normal_decay_iters = 0
        self.response_align_normal_alpha_thresh = 0.9
        self.response_align_normal_neff_thresh = 1.25
        self.response_align_normal_flat_quantile = 0.5
        self.visibility_prune_interval = 0  # 0 = off
        self.visibility_prune_from_iter = 15_000
        self.visibility_prune_until_iter = 30_000
        self.visibility_prune_thresh = 0.05
        self.visibility_prune_max_fraction = 0.0
        # pre-fit subdivision: chop oversized surfels into 4 quadrant children so
        # downstream per-surfel shading (SCIR: one rgb per surfel) has boundaries
        # where shadow gradients need them; schedule BEFORE the end of stage 1 so
        # children settle photometrically.
        self.subdivide_at_iter = 0             # 0 = off
        self.subdivide_scale_mult = 3.0        # chop if smax > mult * median(smax)

        # ---- single-layer densify/prune machinery adaptations ----
        # (1) alpha-preserving clone/split: overlapping children get 1-(1-a)^(1/N) so the
        #     composited alpha (and per-pixel N_eff) is preserved through densification
        #     instead of ~doubling in the footprint at every clone/split.
        self.alpha_preserving_densify = False
        # (3) early transmittance-aware prune during densification (reuses the vprune
        #     max-blend-weight stats): opacity pruning can't see an occluded high-alpha
        #     duplicate layer; its max blend weight can. Fires on its own cadence, skipping
        #     reset iterations and the post-mask0 window where all weights collapse to ~0.01.
        self.contrib_prune_interval = 0        # 0 = off
        self.contrib_prune_thresh = 0.01
        self.contrib_prune_max_fraction = 0.1  # per-call prune cap
        self.contrib_prune_reset_margin = 1500 # min iters past the last mask0 reset
        # (4) dominance-preserving opacity resets: mask0 keeps ray-dominant surfels at their
        #     current opacity (only non-dominant layers are re-opened for competition);
        #     mask1 stops re-inflating non-dominant layers to 0.9.
        self.dominance_reset_thresh = 0.0      # 0 = off; else min max-blend-weight to count as dominant
        self.dominance_reset_from_iter = 15_000  # engage with the ironing phase (= single_warmup_iters)
        # (G) decouple the N_eff loss gradient from the densification accumulator. The N_eff loss
        #     puts large gradients on redundant stacked surfels; the densifier reads those as
        #     "high positional gradient -> clone/split" and amplifies exactly what the loss is
        #     collapsing. Two-backward split (base retain_graph -> stash clean viewspace grad ->
        #     N_eff backward for the step) so densify sees only photometric/geometric gradient.
        self.decouple_single_grad = False
        # Optimizer-revealed capacity routing. Compute the data and N_eff gradients separately;
        # persistent opacity-gradient conflict earns a lagged, budgeted reduction of the complete
        # per-surfel ironing gradient. Disabled by default for exact legacy behavior.
        self.dissent_ironing = False
        self.dissent_beta = 0.98
        self.dissent_tau = 0.25
        self.dissent_min_gate = 0.10
        self.dissent_max_protected_fraction = 0.10
        self.dissent_min_observations = 20
        self.dissent_strength_percentile = 0.90
        self.dissent_min_pressure_ratio = 0.25
        self.dissent_gate_from_iter = 18_000
        self.dissent_reset_margin = 200
        # Continuous alternatives to the default budgeted top-k gate. ``soft``
        # is pure relaxation; ``soft_renorm`` preserves mean consensus pressure
        # by reallocating it onto low-dissent surfels.
        self.dissent_gate_mode = "topk"
        self.dissent_gate_max = 4.0
        # Route the Stage-1 response-alignment loss through the dissent gate alongside the
        # N_eff pressure (protected surfels shielded from both consensus pressures; the
        # data-pressure observation excludes alignment). Off = alignment stays in the data
        # objective, bit-exact with the ungated-alignment runs.
        self.dissent_gate_alignment = False
        # (S) route clone candidates (small, high-grad surfels) through SPLIT instead of clone.
        #     Clone duplicates in place -> a coincident 2nd layer along the ray (N_eff spike the
        #     loss must then undo); split displaces children within the tangent plane (in-surface
        #     refinement, no stacking). Trades clone's size-preserving coverage growth for
        #     single-layer-friendly lateral subdivision (may over-fragment / under-cover detail).
        self.densify_clone_as_split = False

        # ---- PUP-3DGS baseline (Fisher/GN sensitivity pruning, adapted to 2DGS; default off) ----
        # Per-Gaussian score U_g = logdet(H_g), H_g = sum_pixels g_p g_p^T over the SPATIAL
        # params [xyz(3), scaling(2)] -> 5x5 (rotation/opacity/color excluded, faithful to PUP).
        # H_g estimated by a Hutchinson random-projection (unbiased for PUP's exact Fisher):
        # for a +/-1 pixel-sign mask r, one backward of (r*I).sum() gives g = sum_p r_p dI_p/dtheta.
        self.pup_prune = False               # 0 = off, stock behavior bit-exact
        self.pup_prune_iters = [40100]       # iterations at which to prune (list; rounds)
        self.pup_prune_percent = [0.4]       # per-round fraction of the CURRENT count (len == iters)
        self.pup_fisher_draws = 4            # M Hutchinson +/-1 draws per view
        self.pup_fisher_views = -1           # #train views to subsample for scoring; -1 = all
        self.pup_fisher_ridge = 1e-9         # eigenvalue floor for the logdet

        # ---- 2D-SuGaR clustering-prune baseline (spatial truncation; contrast to smooth ironing) ----
        # DBSCAN on surfel centers, keep only the largest connected cluster, drop islands + noise.
        # Enable by passing --cluster_prune_iterations (declared as a nargs list in train_init.py);
        # these scalars mirror 2D-SuGaR's estimate_eps / DBSCAN defaults (min_samples=knn_k=4, p=98).
        self.cluster_prune_eps = 0.0            # 0 = auto (knn-percentile); else fixed DBSCAN eps
        self.cluster_prune_min_samples = 4      # DBSCAN min_samples
        self.cluster_prune_knn_k = 4            # k-th neighbour used for the eps estimate
        self.cluster_prune_knn_percentile = 98.0  # percentile of k-th-NN distances -> eps
        self.cluster_prune_min_cluster_size = 0   # safety floor (0 = off = paper behaviour)

        self.lambda_base_color_smooth = 0.0
        self.lambda_roughness_smooth = 0.0
        self.lambda_metallic_smooth = 0.0
        self.lambda_light_smooth = 0.0

        # initial values
        self.init_roughness_value = 0.1
        self.init_metallic_value = 0.0
        self.init_metallic_value_vol = 0.0
        self.rough_msk_thr = 0.01
        self.metallic_msk_thr = 0.02
        self.metallic_msk_thr_vol = 0.02

        self.enlarge_scale = 1.5


        # Opacity and Densify Settings
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 25000 

        # # Extra settings
        self.densify_grad_threshold = 0.0002
        self.prune_opacity_threshold = 0.05
        self.lambda_light = 0.0  # Weight for light regularization loss
        self.backface_random_from_iter = 50000

        # Stage Settings
        self.initial = 0
        self.init_until_iter = 0 
        self.volume_render_until_iter = 18000 
        self.normal_smooth_from_iter = 0
        self.normal_smooth_until_iter = 18000
                
        self.indirect = 0
        self.indirect_from_iter =  20000 

        self.feature_rest_from_iter = 5_000
        self.normal_prop_until_iter = 25_000 

        self.normal_prop_interval = 1000
        self.opac_lr0_interval = 200
        self.densification_interval_when_prop = 500



        self.normal_loss_start = 0
        self.dist_loss_start = 3000

        # Environmental Scoping
        self.use_env_scope = False
        self.env_scope_center = [0., 0., 0.]
        self.env_scope_radius = 0.0

        # mesh
        self.voxel_size = -1.0
        self.depth_trunc = -1.0
        self.sdf_trunc = -1.0
        self.mesh_res = 512
        self.num_cluster = 1

        # radiosity
        self.train_sh_vol = False
        self.train_sh_surf = False
        self.train_sh_vol_from_iter = 0
        self.train_sh_surf_from_iter = 0
        self.lambda_radiosity = 0.0
        self.radiosity = False
        self.radiosity_from_iter = 50000
        self.reset_sh_features = False

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
