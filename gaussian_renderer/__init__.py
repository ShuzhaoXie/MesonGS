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

import math

import numpy as np
import torch
from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                         GaussianRasterizer,
                                         GaussianRasterizerIndexed)
from torch.autograd import Function

from raht_torch import itransform_batched_torch, transform_batched_torch
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.quant_utils import split_length


def ToEulerAngles_FT(q, save=False):

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.arctan2(sinr_cosp, cosr_cosp)
    
    if save:
        np.save('roll.npy', roll.detach().cpu().numpy())
        np.save('roll_ele.npy', sinr_cosp.detach().cpu().numpy())
        np.save('roll_deno.npy', cosr_cosp.detach().cpu().numpy())

    # pitch (y-axis rotation)
    sinp = torch.sqrt(1 + 2 * (w * y - x * z))
    cosp = torch.sqrt(1 - 2 * (w * y - x * z))
    pitch = 2 * torch.arctan2(sinp, cosp) - torch.pi / 2
    
    if save:
        np.save('pitch.npy', pitch.detach().cpu().numpy())
        np.save('pitch_ele.npy', sinp.detach().cpu().numpy())
        np.save('pitch_deno.npy', cosp.detach().cpu().numpy())
    
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.arctan2(siny_cosp, cosy_cosp)
    
    if save:
        np.save('yaw.npy', yaw.detach().cpu().numpy())
        np.save('yaw_ele.npy', siny_cosp.detach().cpu().numpy())
        np.save('yaw_deno.npy', siny_cosp.detach().cpu().numpy())

    roll = roll.reshape(-1, 1).nan_to_num_()
    pitch = pitch.reshape(-1, 1).nan_to_num_()
    yaw = yaw.reshape(-1, 1).nan_to_num_()

    return torch.concat([roll, pitch, yaw], -1)


class FakeEuler(Function):
    def forward(self, x):
        x = ToEulerAngles_FT(x)
        return x
    
    def backward(self, grad_opt):
        
        return grad_opt


def render(viewpoint_camera, 
           pc : GaussianModel, 
           pipe, 
           bg_color : torch.Tensor, 
           scaling_modifier = 1.0, 
           override_color = None,
           debug=False,
           clamp_color=True,
           meson_count=False,
           f_count=False,
           depth_count=False
           ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=debug,
        clamp_color=clamp_color,
        meson_count=meson_count,
        f_count=f_count,
        depth_count=depth_count
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz
    means2D = screenspace_points

    opacity = pc.get_opacity
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python or pc.get_cov.shape[0] > 0 or pc.get_euler.shape[0] > 0:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
        # print('gaussian_renderer __init__', cov3D_precomp.shape)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # print('shs_view', shs_view.max(), shs_view.min())
            # print('active_sh_degree', pc.active_sh_degree)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # print('sh2rgb.max(), sh2rgb.min()', sh2rgb.max(), sh2rgb.min())
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            # colors_precomp = colors_precomp.nan_to_num(0)
            # colors_precomp = 
            # print('colors_precomp', colors_precomp.max(), colors_precomp.min(), colors_precomp[:5])
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if meson_count:
        rendered_image, radii, imp = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "imp": imp}
    elif f_count:
        rendered_image, radii, imp, gaussians_count, opa_imp = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "imp": imp,
            "gaussians_count": gaussians_count,
            "opa_imp": opa_imp}

    elif depth_count:
        rendered_image, radii, out_depth = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": out_depth}
    else:
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}


def seg_quant(x, lseg, qas):
    lx = x.shape[0]
    cnt = 0
    outs = []
    for i in range(0, lx, lseg):
        if i + lseg < lx:
            r = i + lseg 
        else:
            r = lx
        outs.append(qas[cnt](x[i:r]))
        cnt+=1
    return torch.concat(outs, dim=0)

def seg_quant_ave(x, split, qas):
    start = 0
    cnt = 0
    outs = []
    for length in split:
        outs.append(qas[cnt](x[start:start+length]))
        cnt += 1
        start += length
    return torch.concat(outs, dim=0)


def ft_render(
        viewpoint_camera, 
        pc : GaussianModel, 
        pipe, 
        bg_color : torch.Tensor, 
        training: bool,
        scaling_modifier = 1.0, 
        override_color = None, 
        raht=False, 
        per_channel_quant=False,
        per_block_quant=False,
        debug=False,
        clamp_color=True,
        ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=debug,
        clamp_color=clamp_color,
        meson_count=False,
        f_count=False,
        depth_count=False
    )

    
    rasterizer = GaussianRasterizerIndexed(raster_settings=raster_settings)
    
    # if training:
    #     if pipe.train_mode == 'rot':
    #         re_mode = 'rot'
    #     elif pipe.train_mode == 'euler':
    #         re_mode = 'euler'
    # else:
    #     if pipe.test_mode == 'rot':
    #         re_mode = 'rot'
    #     elif pipe.test_mode == 'euler':
    #         re_mode = 'euler'
    
    
    # if re_mode == 'rot':
    #     re_range = [1, 5]
    #     shzero_range = [5, 8]
    # elif re_mode == 'euler':
    re_range = [1, 4]
    shzero_range = [4, 7]
    
    means3D = pc.get_xyz
    means2D = screenspace_points


    if raht:
        r = pc.get_ori_rotation
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
        q = r / norm[:, None]
        eulers = ToEulerAngles_FT(q, save=False)
        rf = torch.concat([pc.get_origin_opacity, eulers, pc.get_features_dc.contiguous().squeeze()], -1)

        C = rf[pc.reorder]
        iW1 = pc.res['iW1']
        iW2 = pc.res['iW2']
        iLeft_idx = pc.res['iLeft_idx']
        iRight_idx = pc.res['iRight_idx']

        for d in range(pc.depth * 3):
            w1 = iW1[d]
            w2 = iW2[d]
            left_idx = iLeft_idx[d]
            right_idx = iRight_idx[d]
            C[left_idx], C[right_idx] = transform_batched_torch(w1, 
                                                  w2, 
                                                  C[left_idx], 
                                                  C[right_idx])
        
        quantC = torch.zeros_like(C)
        quantC[0] = C[0]
        if per_channel_quant:
            for i in range(C.shape[-1]):
                quantC[1:, i] = pc.qas[i](C[1:, i])
        elif per_block_quant:
            qa_cnt = 0
            lc1 = C.shape[0] - 1
            split_ac = split_length(lc1, pc.n_block)
            for i in range(C.shape[-1]):
                quantC[1:, i] = seg_quant_ave(C[1:, i], split_ac, pc.qas[qa_cnt : qa_cnt + pc.n_block])
                qa_cnt += pc.n_block
            
        else:
            quantC[1:] = pc.qa(C[1:])

        res_inv = pc.res_inv
        pos = res_inv['pos']
        iW1 = res_inv['iW1']
        iW2 = res_inv['iW2']
        iS = res_inv['iS']
        
        iLeft_idx = res_inv['iLeft_idx']
        iRight_idx = res_inv['iRight_idx']
    
        iLeft_idx_CT = res_inv['iLeft_idx_CT']
        iRight_idx_CT = res_inv['iRight_idx_CT']
        iTrans_idx = res_inv['iTrans_idx']
        iTrans_idx_CT = res_inv['iTrans_idx_CT'] 

        CT_yuv_q_temp = quantC[pos.astype(int)]
        raht_features = torch.zeros(quantC.shape).cuda()
        OC = torch.zeros(quantC.shape).cuda()
        
        for i in range(pc.depth*3):
            w1 = iW1[i]
            w2 = iW2[i]
            S = iS[i]
            
            left_idx, right_idx = iLeft_idx[i], iRight_idx[i]
            left_idx_CT, right_idx_CT = iLeft_idx_CT[i], iRight_idx_CT[i]
            
            trans_idx, trans_idx_CT = iTrans_idx[i], iTrans_idx_CT[i]
            
            
            OC[trans_idx] = CT_yuv_q_temp[trans_idx_CT]
            OC[left_idx], OC[right_idx] = itransform_batched_torch(w1, 
                                                    w2, 
                                                    CT_yuv_q_temp[left_idx_CT], 
                                                    CT_yuv_q_temp[right_idx_CT])  
            CT_yuv_q_temp[:S] = OC[:S]

        raht_features[pc.reorder] = OC
        
        scales = pc.get_ori_scaling
        
        if per_channel_quant:
            scalesq = torch.zeros_like(scales).cuda()
            scaleqa_offset = 7
            for i in range(scaleqa_offset, scaleqa_offset + 3):
                scalesq[:, i-scaleqa_offset] = pc.qas[i](scales[:, i-scaleqa_offset])

        elif per_block_quant:
            scalesq = torch.zeros_like(scales).cuda()
            split_scale = split_length(scales.shape[0], pc.n_block)
            for i in range(scales.shape[-1]):
                # scalesq[:, i] = seg_quant(scales[:, i], pc.lseg, pc.qas[qa_cnt : qa_cnt + blocks_in_channel])
                scalesq[:, i] = seg_quant_ave(scales[:, i], split_scale, pc.qas[qa_cnt: qa_cnt + pc.n_block])
                qa_cnt += pc.n_block
        else:
            scalesq = pc.scale_qa(scales)
                
        scaling = torch.exp(scalesq)
        
        # if re_mode == 'rot':
        #     rotations = raht_features[:, 1:5]
        #     cov3D_precomp = pc.covariance_activation(scaling, 1.0, rotations)
        # elif re_mode == 'euler':
        eulers = raht_features[:, 1:4]
        cov3D_precomp = pc.covariance_activation_for_euler(scaling, 1.0, eulers)

        assert cov3D_precomp is not None
        
        opacity = raht_features[:, :1]
        opacity = torch.sigmoid(opacity)    
        
        scales = None
        rotations = None
        eulers = None
        colors_precomp = None
        
        if pipe.use_indexed:
            sh_zero = raht_features[:, shzero_range[0]:].unsqueeze(1).contiguous()
            sh_ones = pc.get_features_extra.reshape(-1, (pc.max_sh_degree+1)**2 - 1, 3)
            sh_indices = pc.get_feature_indices
        else:
            features_dc = raht_features[:, shzero_range[0]:].unsqueeze(1)
            feature_extra = pc.get_indexed_feature_extra
            features = torch.cat((features_dc, feature_extra), dim=1)
            shs_view = features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        raise Exception("Sorry, w/o raht version is unimplemented.")
        
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        opacities = opacity,
        sh_indices = sh_indices,
        sh_zero = sh_zero,
        sh_ones = sh_ones,
        colors_precomp = colors_precomp,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
