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
        self.visibility_prune_interval = 0  # 0 = off
        self.visibility_prune_from_iter = 15_000
        self.visibility_prune_until_iter = 30_000
        self.visibility_prune_thresh = 0.05
        self.visibility_prune_max_fraction = 0.0

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
        # (S) route clone candidates (small, high-grad surfels) through SPLIT instead of clone.
        #     Clone duplicates in place -> a coincident 2nd layer along the ray (N_eff spike the
        #     loss must then undo); split displaces children within the tangent plane (in-surface
        #     refinement, no stacking). Trades clone's size-preserving coverage growth for
        #     single-layer-friendly lateral subdivision (may over-fragment / under-cover detail).
        self.densify_clone_as_split = False

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
