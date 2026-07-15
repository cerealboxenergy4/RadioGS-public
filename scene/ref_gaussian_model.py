import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from scene.light import EnvLight, EnvLightMip
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, rgb_to_srgb
from utils.general_utils import strip_symmetric, build_scaling_rotation, safe_normalize, flip_align_view
import trimesh
import raytracing

def get_env_direction1(H, W):
    gy, gx = torch.meshgrid(torch.linspace(0.0 + 1.0 / H, 1.0 - 1.0 / H, H, device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device='cuda'),
                            indexing='ij')
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    env_directions = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return env_directions


def get_env_direction2(H, W):
    gx, gy = torch.meshgrid(
        torch.linspace(torch.pi, -torch.pi, W, device='cuda'),
        torch.linspace(0, torch.pi, H, device='cuda'),
        indexing='xy'
    )
    env_directions = torch.stack((
        torch.sin(gy)*torch.cos(gx), 
        torch.sin(gy)*torch.sin(gx), 
        torch.cos(gy)
    ), dim=-1)
    return env_directions


class RefGaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.metallic_ativation = torch.sigmoid
        self.inverse_metallic_activation = inverse_sigmoid
        
        self.base_color_activation = torch.sigmoid
        self.inverse_base_color_activation = inverse_sigmoid
        
        self.roughness_activation = torch.sigmoid
        self.inverse_roughness_activation = inverse_sigmoid
        
        self.color_activation = torch.sigmoid
        self.inverse_color_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._base_color = torch.empty(0) 
        self._metallic = torch.empty(0) 
        self._roughness = torch.empty(0) 
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._indirect_dc = torch.empty(0)
        self._indirect_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.radiosity_accum = torch.empty(0)
        self.denom = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.init_metallic_value = 0.01
        self.init_roughness_value = 0.5
        self.init_base_color_value = 0.5
        self.enlarge_scale = 1.5
        self.metallic_msk_thr = 0.02
        self.rough_msk_thr = 0.1

        self.env_map_1 = None
        self.env_map_2 = None
        self.env_H, self.env_W = 256, 512
        self.env_directions1 = get_env_direction1(self.env_H, self.env_W)
        self.env_directions2 = get_env_direction2(self.env_H, self.env_W)
        self.ray_tracer = None
        self.setup_functions()
        
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._metallic, 
            self._roughness, 
            self._base_color, 
            self._features_dc,
            self._features_rest,
            self._indirect_dc,
            self._indirect_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.radiosity_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.env_map_1.state_dict(),
            self.env_map_2.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args=None):
        (self.active_sh_degree, 
        self._xyz, 
        self._metallic, 
        self._roughness, 
        self._base_color, 
        self._features_dc, 
        self._features_rest,
        self._indirect_dc, 
        self._indirect_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum,
        radiosity_accum,
        denom,
        opt_dict, 
        env_1_dict,
        env_2_dict,
        self.spatial_lr_scale) = model_args
        self.env_map_1.load_state_dict(env_1_dict)
        self.env_map_2.load_state_dict(env_2_dict)
        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.radiosity_accum = radiosity_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

    def set_opacity_lr(self, lr):   
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "opacity":
                param_group['lr'] = lr

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) 
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_metallic(self): 
        return self.metallic_ativation(self._metallic)

    @property
    def get_rough(self): 
        return self.roughness_activation(self._roughness)

    @property
    def get_base_color(self): 
        return self.base_color_activation(self._base_color)
    
    def get_normal(self, scaling_modifier, dir_pp_normalized): 
        splat2world = self.get_covariance(scaling_modifier)
        normals_raw = splat2world[:,2,:3] 
        # normals_raw, positive = flip_align_view(normals_raw, dir_pp_normalized)
        normals = safe_normalize(normals_raw)
        
        # R = build_rotation(self._rotation)
        # normals = R[:, :, 2]
        return normals

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_indirect(self):
        indirect_dc = self._indirect_dc
        indirect_rest = self._indirect_rest
        return torch.cat((indirect_dc, indirect_rest), dim=1)
    

    @property
    def get_xyz_gradient_accum(self):
        return self.xyz_gradient_accum / (self.denom + 1e-8)
    
    @property
    def get_radiosity_accum(self):
        return self.radiosity_accum / (self.denom + 1e-8)
    
    def render_env_map_1(self, H=512):
        if H == self.env_H:
            directions1 = self.env_directions1
            directions2 = self.env_directions2
        else:
            W = H * 2
            directions1 = get_env_direction1(H, W)
            directions2 = get_env_direction2(H, W)
        return {'env1': self.env_map_1(directions1, mode="pure_env"), 'env2': self.env_map_1(directions2, mode="pure_env")}
    
    def render_env_map_2(self, H=512):
        if H == self.env_H:
            directions1 = self.env_directions1
            directions2 = self.env_directions2
        else:
            W = H * 2
            directions1 = get_env_direction1(H, W)
            directions2 = get_env_direction2(H, W)
        return {'env1': self.env_map_2(directions1, mode="pure_env"), 'env2': self.env_map_2(directions2, mode="pure_env")}

    @property   
    def get_envmap_1(self): 
        return self.env_map_1
    
    @property   
    def get_envmap_2(self): 
        return self.env_map_2
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, args):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        sh_features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        sh_features[:, :3, 0 ] = fused_color
        sh_features[:, 3:, 1:] = 0.0
        sh_indirect = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        
        def initialize_base_color(point_cloud, init_color= 0.5, noise_level=0.05):
            base_color = torch.full((point_cloud.shape[0], 3), init_color, dtype=torch.float, device="cuda")
            noise = (torch.rand(point_cloud.shape[0], 3, dtype=torch.float, device="cuda") - 0.5) * noise_level
            base_color = base_color + noise
            base_color = torch.clamp(base_color, 0.0, 1.0)
            return base_color
        
        base_color = self.inverse_base_color_activation(initialize_base_color(fused_point_cloud))
        metallic = self.inverse_metallic_activation(torch.full_like(opacities, self.init_metallic_value))
        roughness = self.inverse_roughness_activation(torch.full_like(opacities,  self.init_roughness_value))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

        self._base_color = nn.Parameter(base_color.requires_grad_(True)) 
        self._roughness = nn.Parameter(roughness.requires_grad_(True)) 
        self._metallic = nn.Parameter(metallic.requires_grad_(True)) 
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._features_dc = nn.Parameter(sh_features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(sh_features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_dc = nn.Parameter(sh_indirect[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_rest = nn.Parameter(sh_indirect[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        
        self.env_map_1 = EnvLightMip(path=None, device='cuda', max_res=args.envmap_resolution, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, activation=args.envmap_activation).cuda()
        self.env_map_2 = EnvLightMip(path=None, device='cuda', max_res=args.envmap_resolution, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, activation=args.envmap_activation).cuda()

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.radiosity_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.features_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.features_lr / 20.0, "name": "f_rest"},
            
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.env_map_1.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env1"},
            {'params': self.env_map_2.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env2"}     
        ]

        l.extend([
            {'params': [self._base_color], 'lr': training_args.base_color_lr, "name": "base_color"},  
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},  
            {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"},  
            {'params': [self._indirect_dc], 'lr': training_args.indirect_lr, "name": "ind_dc"},
            {'params': [self._indirect_rest], 'lr': training_args.indirect_lr / 20.0, "name": "ind_rest"},
        ])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._indirect_dc.shape[1]*self._indirect_dc.shape[2]):
            l.append('ind_dc_{}'.format(i))
        for i in range(self._indirect_rest.shape[1]*self._indirect_rest.shape[2]):
            l.append('ind_rest_{}'.format(i))
        l.append('opacity')
        l.append('metallic') 
        l.append('roughness') 
        for i in range(self._base_color.shape[1]):
            l.append('base_color_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ind_dc = self._indirect_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ind_rest = self._indirect_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        metallic = self._metallic.detach().cpu().numpy()    
        roughness = self._roughness.detach().cpu().numpy()    
        base_color = self._base_color.detach().cpu().numpy()    
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, f_rest, ind_dc, ind_rest, opacities, metallic, roughness, base_color, scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        if self.env_map_1 is not None:
            save_path = path.replace('.ply', '_1.map')
            torch.save(self.env_map_1.state_dict(), save_path)
        if self.env_map_2 is not None:
            save_path = path.replace('.ply', '_2.map')
            torch.save(self.env_map_2.state_dict(), save_path)

    def reset_opacity_mask0(self, keep_msk = None):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        if keep_msk is not None:
            # single-layer dominance-preserving reset: the ray-dominant layer keeps its
            # opacity; only non-dominant (duplicate/fringe) layers are re-opened for
            # competition and left to the post-reset opacity prune if nothing revives them.
            opacities_new[keep_msk] = self._opacity[keep_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_mask1(self, exclusive_msk = None, dominant_msk = None):
        RESET_V = 0.9
        opacity_old = self.get_opacity
        o_msk = (opacity_old > RESET_V).flatten()
        if exclusive_msk is not None:
            o_msk = torch.logical_or(o_msk, exclusive_msk)
        if dominant_msk is not None:
            # single-layer: don't re-inflate non-dominant layers to 0.9 — stock behavior
            # revives exactly the duplicate layers the ironing loss ground down.
            o_msk = torch.logical_or(o_msk, ~dominant_msk)
        opacities_new = torch.ones_like(opacity_old)*inverse_sigmoid(torch.tensor([RESET_V]).cuda())
        opacities_new[o_msk] = self._opacity[o_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" not in optimizable_tensors: return
        self._opacity = optimizable_tensors["opacity"]

    def reset_metallic_mask(self, exclusive_msk = None):
        metallic_new = inverse_sigmoid(torch.max(self.get_metallic, torch.ones_like(self.get_metallic)*self.init_metallic_value))
        if exclusive_msk is not None:
            metallic_new[exclusive_msk] = self._metallic[exclusive_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(metallic_new, "metallic")
        if "metallic" not in optimizable_tensors: return
        self._metallic = optimizable_tensors["metallic"]

    def dist_color(self, exclusive_msk = None):
        METALLIC_MSK_THR = self.metallic_msk_thr
        DIST_RANGE = 0.4
        metallic_msk = self.get_metallic.flatten() > METALLIC_MSK_THR
        if exclusive_msk is not None:
            metallic_msk = torch.logical_or(metallic_msk, exclusive_msk)
        dcc = self._features_dc.clone()
        dist_dcc = dcc + (torch.rand_like(dcc)*DIST_RANGE*2-DIST_RANGE) 
        dist_dcc[metallic_msk] = dcc[metallic_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_dcc, "f_dc")
        if "f_dc" not in optimizable_tensors: return
        self._features_dc = optimizable_tensors["f_dc"]

    def enlarge_metallic_scales(self, ret_raw=True, ENLARGE_SCALE=1.5, METALLIC_MSK_THR=0.02, ROUGH_MSK_THR=0.1, exclusive_msk=None):
        ENLARGE_SCALE = self.enlarge_scale
        METALLIC_MSK_THR = self.metallic_msk_thr
        ROUGH_MSK_THR = self.rough_msk_thr

        metallic_msk = self.get_metallic.flatten() < METALLIC_MSK_THR
        rough_msk = self.get_rough.flatten() > ROUGH_MSK_THR
        combined_msk = torch.logical_or(metallic_msk, rough_msk)
        if exclusive_msk is not None:
            combined_msk = torch.logical_or(combined_msk, exclusive_msk) 
        scales = self.get_scaling
        rmin_axis = (torch.ones_like(scales) * ENLARGE_SCALE)
        if ret_raw:
            scale_new = self.scaling_inverse_activation(scales * rmin_axis)
            scale_new[combined_msk] = self._scaling[combined_msk]
        else:
            scale_new = scales * rmin_axis
            scale_new[combined_msk] = scales[combined_msk]   
        return scale_new

    def reset_scale(self, exclusive_msk = None):
        scale_new = self.enlarge_metallic_scales(ret_raw=True, exclusive_msk=exclusive_msk)
        optimizable_tensors = self.replace_tensor_to_optimizer(scale_new, "scaling")
        if "scaling" not in optimizable_tensors: return
        self._scaling = optimizable_tensors["scaling"]


    def reset_features(self, reset_value_dc=0.0, reset_value_rest=0.0):
        features_dc_new = torch.full_like(self._features_dc, reset_value_dc, dtype=torch.float, device="cuda")
        features_rest_new = torch.full_like(self._features_rest, reset_value_rest, dtype=torch.float, device="cuda")

        optimizable_tensors = self.replace_tensor_to_optimizer(features_dc_new, "f_dc")
        optimizable_tensors.update(self.replace_tensor_to_optimizer(features_rest_new, "f_rest"))
        self.active_sh_degree = 0

        if "f_dc" in optimizable_tensors:
            self._features_dc = optimizable_tensors["f_dc"]
        if "f_rest" in optimizable_tensors:
            self._features_rest = optimizable_tensors["f_rest"]


    def reset_base_color(self, reset_value=0.5, noise_level=0.05):
        base_color = torch.full_like(self._base_color, reset_value, dtype=torch.float, device="cuda")
        noise = (torch.rand_like(base_color, dtype=torch.float, device="cuda") - 0.5) * noise_level
        base_color_new = base_color + noise
        base_color_new = torch.clamp(base_color_new, 0.0, 1.0)
        
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_color_activation(base_color_new), "base_color")
        if "base_color" in optimizable_tensors:
            self._base_color = optimizable_tensors["base_color"]

    def reset_metallic(self, reset_value=0.01):
        metallic_new = torch.full_like(self._metallic, reset_value, dtype=torch.float32, device="cuda")
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_metallic_activation(metallic_new), "metallic")
        if "metallic" in optimizable_tensors:
            self._metallic = optimizable_tensors["metallic"]
    
    def reset_roughness(self, reset_value=0.1):
        roughness_new = torch.full_like(self._roughness, reset_value, dtype=torch.float, device="cuda")
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_roughness_activation(roughness_new), "roughness")
        if "roughness" in optimizable_tensors:
            self._roughness = optimizable_tensors["roughness"]

    def load_ply(self, path, relight=False, envmap_activation='sigmoid'):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        base_color = np.stack((np.asarray(plydata.elements[0]['base_color_0']),
                              np.asarray(plydata.elements[0]['base_color_1']),
                              np.asarray(plydata.elements[0]['base_color_2'])),  axis=1)
        
        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis] # #
        metallic = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis] # #

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        self.active_sh_degree = self.max_sh_degree
        
        indirect_dc = np.zeros((xyz.shape[0], 3, 1))
        indirect_dc[:, 0, 0] = np.asarray(plydata.elements[0]["ind_dc_0"])
        indirect_dc[:, 1, 0] = np.asarray(plydata.elements[0]["ind_dc_1"])
        indirect_dc[:, 2, 0] = np.asarray(plydata.elements[0]["ind_dc_2"])

        extra_ind_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ind_rest_")]
        extra_ind_names = sorted(extra_ind_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_ind_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        indirect_extra = np.zeros((xyz.shape[0], len(extra_ind_names)))
        for idx, attr_name in enumerate(extra_ind_names):
            indirect_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        indirect_extra = indirect_extra.reshape((indirect_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        map_path1 = path.replace('.ply', '_1.map')
        if os.path.exists(map_path1):
            map_ckpt = torch.load(map_path1)
            self.env_map_1 = EnvLightMip(path=None, device='cuda', max_res=map_ckpt['base'].shape[1], activation=envmap_activation).cuda()
            self.env_map_1.load_state_dict(map_ckpt)

        map_path2 = path.replace('.ply', '_2.map')
        if os.path.exists(map_path2):
            map_ckpt = torch.load(map_path2)
            self.env_map_2 = EnvLightMip(path=None, device='cuda', max_res=map_ckpt['base'].shape[1], activation=envmap_activation).cuda()
            self.env_map_2.load_state_dict(map_ckpt)
            
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._metallic = nn.Parameter(torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._base_color = nn.Parameter(torch.tensor(base_color, dtype=torch.float, device="cuda").requires_grad_(True))   # #

        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self._indirect_dc = nn.Parameter(torch.tensor(indirect_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_rest = nn.Parameter(torch.tensor(indirect_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is None: continue
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "env1" or group["name"] == "env2": continue   # #
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]

        self._base_color = optimizable_tensors['base_color']    # #
        self._roughness = optimizable_tensors['roughness']    # #
        self._metallic = optimizable_tensors['metallic']    # #

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._indirect_dc = optimizable_tensors["ind_dc"]
        self._indirect_rest = optimizable_tensors["ind_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.radiosity_accum = self.radiosity_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        # single-layer: carry contribution stats through pruning so the dominance-preserving
        # resets and the early contrib prune (which fire on densify iterations, after the
        # point count already changed) still see aligned stats instead of a forced reset.
        if getattr(self, "contribution_accum", torch.empty(0)).shape[0] == valid_points_mask.shape[0]:
            self.contribution_accum = self.contribution_accum[valid_points_mask]
            self.contribution_view_count = self.contribution_view_count[valid_points_mask]

    # ---- single-layer visibility prune (ported from single-layer-surfel Part 1) ----
    def reset_contribution_stats(self):
        n = self.get_xyz.shape[0]
        self.contribution_accum = torch.zeros(n, device="cuda")
        self.contribution_view_count = torch.zeros(n, device="cuda")

    def add_contribution_stats(self, surfel_contrib, visibility_filter=None):
        if getattr(self, "contribution_accum", torch.empty(0)).shape[0] != self.get_xyz.shape[0]:
            self.reset_contribution_stats()
        self.contribution_accum = torch.maximum(self.contribution_accum, surfel_contrib.detach())
        if visibility_filter is not None:
            self.contribution_view_count[visibility_filter] += 1

    def get_dominant_mask(self, thresh):
        """Surfels whose max blend weight since the last stats reset clears `thresh` (w > 0.5 on
        some pixel implies argmax there). None if stats are missing or misaligned."""
        accum = getattr(self, "contribution_accum", None)
        if accum is None or accum.shape[0] != self.get_xyz.shape[0]:
            return None
        return accum >= thresh

    def visibility_prune(self, contribution_threshold, min_visible_views=1, max_prune_fraction=0.0):
        """Prune Gaussians whose max per-pixel contribution across all seen views stays below
        contribution_threshold (seen in >= min_visible_views), capped at max_prune_fraction of points."""
        if getattr(self, "contribution_accum", torch.empty(0)).shape[0] != self.get_xyz.shape[0]:
            self.reset_contribution_stats()
            return 0
        n = self.get_xyz.shape[0]
        eligible = self.contribution_view_count >= min_visible_views
        candidate = eligible & (self.contribution_accum < contribution_threshold)
        if max_prune_fraction and candidate.sum() > 0:
            cap = max(1, int(n * max_prune_fraction))
            if int(candidate.sum().item()) > cap:
                idx = torch.nonzero(candidate, as_tuple=False).flatten()
                keep = self.contribution_accum[idx].argsort()[:cap]  # prune the lowest-contribution first
                new_mask = torch.zeros_like(candidate)
                new_mask[idx[keep]] = True
                candidate = new_mask
        pruned = int(candidate.sum().item())
        if pruned > 0:
            self.prune_points(candidate)
            self.reset_contribution_stats()  # sizes changed; rebuild
        print(f"[vprune] thr={contribution_threshold} eligible={int(eligible.sum().item())} "
              f"pruned={pruned} -> {n - pruned}/{n} kept", flush=True)
        return pruned

    # ---- 2D-SuGaR clustering prune (baseline: spatial truncation, contrast to smooth ironing) ----
    def cluster_prune(self, eps=0.0, min_samples=4, knn_k=4, knn_percentile=98.0, min_cluster_size=0):
        """DBSCAN clustering prune ported faithfully from 2D-SuGaR (Divyam et al., EG'26),
        gaussian_splatting_2d/scene/gaussian_model.py::cluster_gaussians + the train.py invocation:
        run DBSCAN on the surfel centers, keep ONLY the largest connected cluster, and prune every
        other cluster plus all DBSCAN noise points (mask = labels != argmax-cluster). eps is
        auto-estimated (when eps<=0) as the knn_percentile-th percentile of each point's distance to
        its knn_k-th nearest neighbour (the paper's estimate_eps; defaults min_samples=knn_k=4,
        percentile=98). This is a hard *spatial* truncation baseline: unlike single-layer ironing
        (a smooth per-ray N_eff overlap penalty) it removes whole primitives by spatial connectivity.
        Returns the number of surfels pruned."""
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
        n = self.get_xyz.shape[0]
        pts = self.get_xyz.detach().cpu().numpy()
        if eps is None or eps <= 0.0:
            k = max(2, int(knn_k))
            nbrs = NearestNeighbors(n_neighbors=k).fit(pts)
            dists, _ = nbrs.kneighbors(pts)
            eps = float(np.percentile(dists[:, -1], knn_percentile))  # distance to the k-th neighbour
        labels = DBSCAN(eps=eps, min_samples=int(min_samples)).fit_predict(pts)
        uniq, counts = np.unique(labels[labels >= 0], return_counts=True)  # non-noise clusters only
        if uniq.size == 0:  # everything is noise -> "largest cluster" undefined; no-op (paper would crash)
            print(f"[cluster-prune] eps={eps:.5f} clusters=0 noise={n} -> no-op, kept {n}/{n}", flush=True)
            return 0
        retained = int(uniq[int(np.argmax(counts))])
        retained_size = int(counts.max())
        # safety floor (default off, = paper behaviour): never amputate the object if even the biggest
        # cluster is smaller than min_cluster_size.
        if min_cluster_size > 0 and retained_size < min_cluster_size:
            print(f"[cluster-prune] eps={eps:.5f} largest cluster {retained_size} < min "
                  f"{min_cluster_size} -> no-op, kept {n}/{n}", flush=True)
            return 0
        mask = torch.from_numpy(labels != retained).to(self.get_xyz.device)
        pruned = int(mask.sum().item())
        if pruned > 0:
            self.prune_points(mask)
            if getattr(self, "contribution_accum", torch.empty(0)).shape[0] != self.get_xyz.shape[0]:
                self.reset_contribution_stats()  # sizes changed; rebuild vprune stats
        print(f"[cluster-prune] eps={eps:.5f} clusters={int(uniq.size)} retained={retained} "
              f"(size {retained_size}) noise={int((labels < 0).sum())} "
              f"pruned={pruned} -> {n - pruned}/{n} kept", flush=True)
        return pruned

    def subdivide_large(self, scale_mult=3.0, max_frac=0.2):
        """Chop oversized surfels into 4 quadrant children in the splat plane
        (offsets +-0.5*s_u*tu +-0.5*s_v*tv, scale/2, inherited attrs). One surfel
        shades to ONE rgb downstream, so shadow gradients need surfel boundaries;
        textureless regions (plates) never densify from image gradients."""
        smax = torch.max(self.get_scaling, dim=1).values
        thresh = scale_mult * smax.median()
        mask = smax > thresh
        cap = int(max_frac * smax.shape[0])
        if int(mask.sum()) > cap:
            keep = torch.topk(smax, cap).indices
            mask = torch.zeros_like(mask)
            mask[keep] = True
        n = int(mask.sum())
        if n == 0:
            return 0
        R = build_rotation(self._rotation[mask])
        s = self.get_scaling[mask]
        su = s[:, 0:1] * R[:, :, 0]
        sv = s[:, 1:2] * R[:, :, 1]
        xyz_m = self.get_xyz[mask]
        new_xyz = torch.cat([xyz_m + du * su + dv * sv
                             for du, dv in ((0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5))], 0)
        rep = lambda t: t[mask].repeat(4, *([1] * (t.dim() - 1)))
        new_scaling = self.scaling_inverse_activation(s.repeat(4, 1) / 2.0)
        self.densification_postfix(new_xyz, rep(self._metallic), rep(self._roughness),
                                   rep(self._base_color), rep(self._features_dc),
                                   rep(self._features_rest), rep(self._indirect_dc),
                                   rep(self._indirect_rest), rep(self._opacity),
                                   new_scaling, rep(self._rotation))
        prune_mask = torch.cat([mask, torch.zeros(4 * n, dtype=torch.bool, device=mask.device)])
        self.prune_points(prune_mask)
        return n

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "env1" or group["name"] == "env2": continue   # #
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_metallic, new_roughness, new_base_color, new_features_dc, new_features_rest, new_indirect_dc, new_indirect_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "metallic": new_metallic,
        "roughness": new_roughness,
        "base_color": new_base_color,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "ind_dc": new_indirect_dc,
        "ind_rest": new_indirect_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._metallic = optimizable_tensors['metallic']    # #
        self._roughness = optimizable_tensors['roughness']    # #
        self._base_color = optimizable_tensors['base_color']    # #
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._indirect_dc = optimizable_tensors["ind_dc"]
        self._indirect_rest = optimizable_tensors["ind_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.radiosity_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, alpha_preserving=False,
                          include_small=False):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        if not include_small:
            # stock: split only over-reconstructed (large) surfels; small ones go to clone.
            # (S) include_small=True splits the clone candidates too (single-layer lateral refine).
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                  torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_base_color = self._base_color[selected_pts_mask].repeat(N,1)   # #
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)   # #
        new_metallic = self._metallic[selected_pts_mask].repeat(N,1)   # #

        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        
        new_indirect_dc = self._indirect_dc[selected_pts_mask].repeat(N,1,1)
        new_indirect_rest = self._indirect_rest[selected_pts_mask].repeat(N,1,1)
        
        if alpha_preserving:
            # 1-(1-a)^(1/N) per child preserves the composited alpha of the N overlapping
            # children in the parent footprint (stock repeat ~doubles it -> instant extra layer)
            a = self.get_opacity[selected_pts_mask].clamp(1e-6, 1.0 - 1e-6)
            new_opacity = self.inverse_opacity_activation(
                (1.0 - torch.pow(1.0 - a, 1.0 / N)).clamp(1e-4, 1.0 - 1e-4)).repeat(N, 1)
        else:
            new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_metallic, new_roughness, new_base_color, new_features_dc, new_features_rest, new_indirect_dc, new_indirect_rest, new_opacity, new_scaling, new_rotation)

        # single-layer: children inherit the parent's contribution stats (extend BEFORE the
        # parent prune below so prune_points slices an aligned array)
        if getattr(self, "contribution_accum", torch.empty(0)).shape[0] == n_init_points:
            self.contribution_accum = torch.cat(
                [self.contribution_accum, self.contribution_accum[selected_pts_mask].repeat(N)])
            self.contribution_view_count = torch.cat(
                [self.contribution_view_count, self.contribution_view_count[selected_pts_mask].repeat(N)])

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, alpha_preserving=False):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]

        new_metallic = self._metallic[selected_pts_mask]   # #
        new_roughness = self._roughness[selected_pts_mask]   # #
        new_base_color = self._base_color[selected_pts_mask]   # #

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        
        new_indirect_dc = self._indirect_dc[selected_pts_mask]
        new_indirect_rest = self._indirect_rest[selected_pts_mask]
        
        new_opacities = self._opacity[selected_pts_mask]
        if alpha_preserving and bool(selected_pts_mask.any()):
            # a clone is two coincident copies: give BOTH 1-sqrt(1-a) so their composited
            # alpha equals the original's (stock keeps a on both -> alpha and N_eff ~double)
            a = self.get_opacity[selected_pts_mask].clamp(1e-6, 1.0 - 1e-6)
            logits = self.inverse_opacity_activation((1.0 - torch.sqrt(1.0 - a)).clamp(1e-4, 1.0 - 1e-4))
            self._opacity.data[selected_pts_mask] = logits  # the surviving original of the pair
            new_opacities = logits                          # the copy
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_metallic, new_roughness, new_base_color, new_features_dc, new_features_rest, new_indirect_dc, new_indirect_rest, new_opacities, new_scaling, new_rotation)

        # single-layer: cloned copies inherit the parent's contribution stats
        if getattr(self, "contribution_accum", torch.empty(0)).shape[0] == selected_pts_mask.shape[0]:
            self.contribution_accum = torch.cat(
                [self.contribution_accum, self.contribution_accum[selected_pts_mask]])
            self.contribution_view_count = torch.cat(
                [self.contribution_view_count, self.contribution_view_count[selected_pts_mask]])

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, alpha_preserving=False,
                          clone_as_split=False):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if self.get_opacity.shape[0] < 20_000_000:
            if clone_as_split:
                # (S) route BOTH branches through split: small (clone-target) and large surfels
                # are subdivided in-plane instead of the small ones being cloned (stacked) in place.
                self.densify_and_split(grads, max_grad, extent, alpha_preserving=alpha_preserving,
                                       include_small=True)
            else:
                self.densify_and_clone(grads, max_grad, extent, alpha_preserving=alpha_preserving)
                self.densify_and_split(grads, max_grad, extent, alpha_preserving=alpha_preserving)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, radiosity_tensor=None, clean_grad=None):
        # single-layer (G): clean_grad is the viewspace gradient from the photometric/geometric
        # losses only (N_eff term excluded) so densification isn't steered by the ironing loss.
        grad_src = clean_grad if clean_grad is not None else viewspace_point_tensor.grad
        self.xyz_gradient_accum[update_filter] += torch.norm(grad_src[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    def update_mesh(self, mesh):
        vertices = np.asarray(mesh.vertices).astype(np.float32)
        faces = np.asarray(mesh.triangles).astype(np.int32)
        self.ray_tracer = raytracing.RayTracer(vertices, faces)
        