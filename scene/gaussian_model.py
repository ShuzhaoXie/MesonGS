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

import os
import time

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
import shutil
from raht_torch import (copyAsort, haar3D_param,
                        inv_haar3D_param,
                        itransform_batched_torch, transform_batched_torch)
from utils.general_utils import (build_rotation, build_scaling_rotation,
                                 get_expon_lr_func, inverse_sigmoid,
                                 strip_symmetric)
from utils.graphics_utils import BasicPointCloud
from utils.quant_utils import VanillaQuan
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p
from vq import vq_features

def d1halfing_fast(pmin,pmax,pdepht):
    return np.linspace(pmin,pmax,2**int(pdepht)+1)

def ToEulerAngles_FT(q):

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = torch.sqrt(1 + 2 * (w * y - x * z))
    cosp = torch.sqrt(1 - 2 * (w * y - x * z))
    pitch = 2 * torch.arctan2(sinp, cosp) - torch.pi / 2
    
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.arctan2(siny_cosp, cosy_cosp)

    roll = roll.reshape(-1, 1)
    pitch = pitch.reshape(-1, 1)
    yaw = yaw.reshape(-1, 1)

    return torch.concat([roll, pitch, yaw], -1)

def calcScaleZeroPoint(min_val, max_val, num_bits=32):
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    scale = (max_val - min_val) / (qmax - qmin)

    zero_point = qmax - max_val / scale

    if zero_point < qmin:
        zero_point = torch.tensor([qmin], dtype=torch.float32).to(min_val.device)
    elif zero_point > qmax:
        # zero_point = qmax
        zero_point = torch.tensor([qmax], dtype=torch.float32).to(max_val.device)
    
    zero_point.round_()

    return scale, zero_point

def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False):
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2. ** num_bits - 1.
 
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    
    return q_x
 
def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x - zero_point)

def build_rotation_from_euler(roll, pitch, yaw):
    R = torch.zeros((roll.size(0), 3, 3), device='cuda')

    R[:, 0, 0] = torch.cos(pitch) * torch.cos(roll)
    R[:, 0, 1] = -torch.cos(yaw) * torch.sin(roll) + torch.sin(yaw) * torch.sin(pitch) * torch.cos(roll)
    R[:, 0, 2] = torch.sin(yaw) * torch.sin(roll) + torch.cos(yaw) * torch.sin(pitch) * torch.cos(roll)
    R[:, 1, 0] = torch.cos(pitch) * torch.sin(roll)
    R[:, 1, 1] = torch.cos(yaw) * torch.cos(roll) + torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll)
    R[:, 1, 2] = -torch.sin(yaw) * torch.cos(roll) + torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll)
    R[:, 2, 0] = -torch.sin(pitch)
    R[:, 2, 1] = torch.sin(yaw) * torch.cos(pitch)
    R[:, 2, 2] = torch.cos(yaw) * torch.cos(pitch)

    return R
                       
def octreecodes(ppoints, pdepht, merge_type='mean',imps=None):
    minx=np.amin(ppoints[:,0])
    maxx=np.amax(ppoints[:,0])
    miny=np.amin(ppoints[:,1])
    maxy=np.amax(ppoints[:,1])
    minz=np.amin(ppoints[:,2])
    maxz=np.amax(ppoints[:,2])
    xletra=d1halfing_fast(minx,maxx,pdepht)
    yletra=d1halfing_fast(miny,maxy,pdepht)
    zletra=d1halfing_fast(minz,maxz,pdepht)
    otcodex=np.searchsorted(xletra,ppoints[:,0],side='right')-1
    otcodey=np.searchsorted(yletra,ppoints[:,1],side='right')-1
    otcodez=np.searchsorted(zletra,ppoints[:,2],side='right')-1
    ki=otcodex*(2**(pdepht*2))+otcodey*(2**pdepht)+otcodez
    
    ki_ranks = np.argsort(ki)
    ppoints = ppoints[ki_ranks]
    ki = ki[ki_ranks]

    ppoints = np.concatenate([ki.reshape(-1, 1), ppoints], -1)
    # print('here 4', ppoints.shape)
    dedup_points = np.split(ppoints[:, 1:], np.unique(ki, return_index=True)[1][1:])
    
    # print('ki.shape', ki.shape)
    
    # print('ki.shape', ki.shape)
    final_feature = []
    if merge_type == 'mean':
        for dedup_point in dedup_points:
            # print(np.mean(dedup_point, 0).shape)
            final_feature.append(np.mean(dedup_point, 0).reshape(1, -1))
    elif merge_type == 'imp':
        dedup_imps = np.split(imps, np.unique(ki, return_index=True)[1][1:])
        for dedup_point, dedup_imp in zip(dedup_points, dedup_imps):
            dedup_imp = dedup_imp.reshape(1, -1)
            if dedup_imp.shape[-1] == 1:
                # print('dedup_point.shape', dedup_point.shape)
                final_feature.append(dedup_point)
            else:
                # print('dedup_point.shape, dedup_imp.shape', dedup_point.shape, dedup_imp.shape)
                fdp = (dedup_imp / np.sum(dedup_imp)) @ dedup_point
                # print('fdp.shape', fdp.shape)
                final_feature.append(fdp)
    elif merge_type == 'rand':
        for dedup_point in dedup_points:
            ld = len(dedup_point)
            id = torch.randint(0, ld, (1,))[0]
            final_feature.append(dedup_point[id].reshape(1, -1))
    else:
        raise NotImplementedError
    ki = np.unique(ki)
    final_feature = np.concatenate(final_feature, 0)
    # print('final_feature.shape', final_feature.shape)
    return (ki,minx,maxx,miny,maxy,minz,maxz, final_feature)


def create_octree_overall(ppoints, pfeatures, imp, depth, oct_merge):
    ori_points_num = ppoints.shape[0]
    ppoints = np.concatenate([ppoints, pfeatures], -1)
    occ=octreecodes(ppoints, depth, oct_merge, imp)
    final_points_num = occ[0].shape[0]
    occodex=(occ[0]/(2**(depth*2))).astype(int)
    occodey=((occ[0]-occodex*(2**(depth*2)))/(2**depth)).astype(int)
    occodez=(occ[0]-occodex*(2**(depth*2))-occodey*(2**depth)).astype(int)
    voxel_xyz = np.array([occodex,occodey,occodez], dtype=int).T
    features = occ[-1][:, 3:]
    paramarr=np.asarray([occ[1],occ[2],occ[3],occ[4],occ[5],occ[6]]) # boundary
    # print('oct[0]', type(oct[0]))
    return voxel_xyz, features, occ[0], paramarr, ori_points_num, final_points_num

def torch_seg_quant(x, lseg, qas):
    lx = x.shape[0]
    cnt = 0
    outs = []
    trans = []
    for i in range(0, lx, lseg):
        if i + lseg < lx:
            r = i + lseg 
        else:
            r = lx
        i_scale = qas[cnt].scale
        i_zp = qas[cnt].zero_point
        i_dtype = qas[cnt].dtype
        outs.append(torch.quantize_per_tensor(
            x[i:r],
            scale=i_scale,
            zero_point=i_zp,
            dtype=i_dtype
        ).int_repr().cpu().numpy())
        trans.extend([i_scale.item(), i_zp.item()])
        cnt+=1
    return np.concatenate(outs, axis=0), trans

def torch_vanilla_quant(x, lseg, qas):
    lx = x.shape[0]
    cnt = 0
    outs = []
    trans = []
    for i in range(0, lx, lseg):
        if i + lseg < lx:
            r = i + lseg 
        else:
            r = lx
        i_scale = qas[cnt].scale
        i_zp = qas[cnt].zero_point
        i_bit = qas[cnt].bit
        outs.append(quantize_tensor(
            x[i:r],
            scale=i_scale,
            zero_point=i_zp,
            num_bits=i_bit).cpu().numpy())
        trans.extend([i_scale.item(), i_zp.item()])
        cnt+=1
    return np.concatenate(outs, axis=0), trans


def torch_vanilla_dequant(x, lseg, sz):
    lx = x.shape[0]
    cnt = 0 
    outs = []
    for i in range(0, lx, lseg):
        if i + lseg < lx:
            r = i + lseg 
        else:
            r = lx
        i_scale = sz[cnt]
        i_zp = sz[cnt+1]
        outs.append(
            dequantize_tensor(
                x[i:r],
                scale=i_scale,
                zero_point=i_zp
            )
        )
        cnt+=2
    return torch.concat(outs, axis=0)


def decode_oct(paramarr, oct, depth):
    minx=(paramarr[0])
    maxx=(paramarr[1])
    miny=(paramarr[2])
    maxy=(paramarr[3])
    minz=(paramarr[4])
    maxz=(paramarr[5])
    xletra=d1halfing_fast(minx,maxx,depth)
    yletra=d1halfing_fast(miny,maxy,depth)
    zletra=d1halfing_fast(minz,maxz,depth)
    occodex=(oct/(2**(depth*2))).astype(int)
    occodey=((oct-occodex*(2**(depth*2)))/(2**depth)).astype(int)
    occodez=(oct-occodex*(2**(depth*2))-occodey*(2**depth)).astype(int)  
    V = np.array([occodex,occodey,occodez], dtype=int).T
    koorx=xletra[occodex]
    koory=yletra[occodey]
    koorz=zletra[occodez]
    ori_points=np.array([koorx,koory,koorz]).T

    return ori_points, V

def check_nonzero(c):
    for i in c:
        if i < 0:
            return False 
    return True

class GaussianModel:
    def __init__(self, sh_degree : int, depth=12, num_bits=8):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, return_symm=True):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            if return_symm:
                symm = strip_symmetric(actual_covariance)
                return symm
            else:
                return actual_covariance

        def build_covariance_from_scaling_euler(scaling, scaling_modifier, euler, return_symm=True):
            s = scaling_modifier * scaling
            L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
            R = build_rotation_from_euler(euler[:, 2], euler[:, 1], euler[:, 0])

            L[:,0,0] = s[:,0]
            L[:,1,1] = s[:,1]
            L[:,2,2] = s[:,2]

            L = R @ L
            actual_covariance = L @ L.transpose(1, 2)
            if return_symm:
                symm = strip_symmetric(actual_covariance)
                return symm
            else:
                return actual_covariance
            
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._cov = torch.empty(0)
        self._euler = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)

        '''for finetune'''
        self.num_bits = num_bits
        self.depth = depth
        self._V = None
        self.optimizer = None
        self.w = None
        self.val = None
        self.TMP = None
        self.res_tree = None
        self.ret_features = None
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.covariance_activation_for_euler = build_covariance_from_scaling_euler

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_ori_scaling(self):
        return self._scaling
    
    @property
    def get_ori_rotation(self):
        return self._rotation
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        # print("features_dc Requires Grad: ", self._features_dc.requires_grad)
        # print("features_rest Requires Grad: ", self._features_dc.requires_grad)
        return torch.cat((self._features_dc, self._features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_extra(self):
        features_extra = self._features_rest.reshape((-1, 3, (self.max_sh_degree + 1) ** 2 - 1))
        return features_extra
    
    @property
    def get_indexed_feature_extra(self):
        n_sh = (self.active_sh_degree + 1) ** 2
        num_points = self.get_xyz.shape[0]
        fi = self._feature_indices.detach().cpu()
        fr = self._features_rest.detach().cpu()
        ret = torch.zeros([num_points, 3 * (n_sh - 1)])
        for i in range(num_points):
            ret[i] = self._features_rest[int(fi[i])]
        return ret.reshape(-1, n_sh - 1, 3)
        # return torch.matmul(F.one_hot(self._feature_indices).float(), self._features_rest).reshape(-1, self.n_sh - 1, 3)

    @property
    def get_feature_indices(self):
        return self._feature_indices
    
    @property
    def get_opacity(self):
        # print("self._opacity Requires Grad: ", self._opacity.requires_grad)
        return self.opacity_activation(self._opacity)
    
    @property
    def get_origin_opacity(self):
        return self._opacity
    
    @property
    def get_cov(self):
        return self._cov

    @property
    def get_euler(self):
        return self._euler
    
    @property
    def get_V(self):
        return self._V

    def get_covariance(self, scaling_modifier = 1):
        if self.get_euler.shape[0] > 0:
            # print('go with euler')
            return self.covariance_activation_for_euler(self.get_scaling, scaling_modifier, self._euler)
        elif self.get_cov.shape[0] > 0:
            return self.get_cov
        else:
            # print('gaussian model: get cov from scaling and rotations.')
            return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def _check_spd_cov(self, cov):
        cov = cov.detach().cpu().numpy()
        
        eig, _ = np.linalg.eig(cov)
        pdm = []
        for i in range(eig.shape[0]):
            if (check_nonzero(eig[i])):
                pdm.append(True)
            else:
                pdm.append(False)
        pdm = np.array(pdm)
        return pdm
    
    def check_spd(self):
        cov_r = self.covariance_activation(self.get_scaling, 1.0, self._rotation, False)
        # cov_e = self.covariance_activation_for_euler(self.get_scaling, 1.0, self.get_euler, False)
        num = cov_r.shape[0]
        pdm_r = self._check_spd_cov(cov_r)
        # pmd_e = self._check_spd_cov(cov_e)
        print("rotation spd ratio: ", pdm_r.sum() / num)
        return pdm_r

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        print('create_from_pcd fused_point_cloud.shape', fused_point_cloud.shape)
        tmp_pcd_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        fused_color = RGB2SH(tmp_pcd_color)

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        if training_args.finetune_lr_scale < 1.0 - 0.001:
            print('training setup: finetune')
            training_args.position_lr_init = training_args.position_lr_init * training_args.finetune_lr_scale
            training_args.feature_lr = training_args.feature_lr * training_args.finetune_lr_scale
            training_args.opacity_lr = training_args.opacity_lr * training_args.finetune_lr_scale
            training_args.scaling_lr = training_args.scaling_lr * training_args.finetune_lr_scale
            training_args.rotation_lr = training_args.rotation_lr * training_args.finetune_lr_scale
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init*self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr*self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.posititon_lr_max_steps)
    
    def finetuning_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    
        print('finetuning setup: finetune')
        training_args.position_lr_init = training_args.position_lr_init * training_args.finetune_lr_scale
        training_args.feature_lr = training_args.feature_lr * training_args.finetune_lr_scale
        training_args.opacity_lr = training_args.opacity_lr * training_args.finetune_lr_scale
        training_args.scaling_lr = training_args.scaling_lr * training_args.finetune_lr_scale
        training_args.rotation_lr = training_args.rotation_lr * training_args.finetune_lr_scale
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init*self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr*self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.posititon_lr_max_steps)
    

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def ft_construct_list_of_attributes(self, fshape1):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(fshape1):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_npz(self, exp_dir):
        
        os.makedirs(exp_dir, exist_ok=True)
        bin_dir = os.path.join(exp_dir, 'bins')
        os.makedirs(bin_dir, exist_ok=True)
        trans_array = []
        trans_array.append(self.depth)
        trans_array.append(self.lseg)

        with torch.no_grad():
            np.savez_compressed(os.path.join(bin_dir , 'oct'), points=self.oct, params=self.oct_param)

            ntk = self._feature_indices.detach().contiguous().cpu().int().numpy()
            cb = self._features_rest.detach().contiguous().cpu().numpy()
            # print('cb.shape', cb.shape)
            np.savez_compressed(os.path.join(bin_dir , 'ntk.npz'), ntk=ntk)
            np.savez_compressed(os.path.join(bin_dir , 'um.npz'), umap=cb)
            
            r = self.get_ori_rotation
            norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
            q = r / norm[:, None]
            eulers = ToEulerAngles_FT(q)
            
            rf = torch.concat([self.get_origin_opacity.detach(), eulers.detach(), self.get_features_dc.detach().contiguous().squeeze()], axis=-1)
            
            C = rf[self.reorder]
            iW1 = self.res['iW1']
            iW2 = self.res['iW2']
            iLeft_idx = self.res['iLeft_idx']
            iRight_idx = self.res['iRight_idx']

            for d in range(self.depth * 3):
                w1 = iW1[d]
                w2 = iW2[d]
                left_idx = iLeft_idx[d]
                right_idx = iRight_idx[d]
                C[left_idx], C[right_idx] = transform_batched_torch(w1, 
                                                    w2, 
                                                    C[left_idx], 
                                                    C[right_idx])

            cf = C[0].cpu().numpy()

            qa_cnt = 0
            lc1 = C.shape[0] - 1
            qci = [] 
            if lc1 % self.lseg == 0:
                blocks_in_channel = lc1 // self.lseg
            else:
                blocks_in_channel = lc1 // self.lseg + 1
            for i in range(C.shape[-1]):
                t1, trans1 = torch_vanilla_quant(C[1:, i], self.lseg, self.qas[qa_cnt : qa_cnt + blocks_in_channel])
                qci.append(t1)
                # .reshape(-1, 1)
                trans_array.extend(trans1)
                qa_cnt += blocks_in_channel
            qci = np.concatenate(qci, axis=-1)
                
            np.savez_compressed(os.path.join(bin_dir,'orgb.npz'), f=cf, i=qci.astype(np.uint8))
            
        
            scaling = self.get_ori_scaling.detach()

            lc1 = scaling.shape[0]
            scaling_q = []
            if lc1 % self.lseg == 0:
                blocks_in_channel = lc1 // self.lseg
            else:
                blocks_in_channel = lc1 // self.lseg + 1
            for i in range(scaling.shape[-1]):
                t1, trans1 = torch_vanilla_quant(scaling[:, i], self.lseg, self.qas[qa_cnt : qa_cnt + blocks_in_channel])
                scaling_q.append(t1)
                # .reshape(-1, 1)
                trans_array.extend(trans1)
                qa_cnt += blocks_in_channel
            scaling_q = np.concatenate(scaling_q, axis=-1)

            np.savez_compressed(os.path.join(bin_dir,'ct.npz'), i=scaling_q.astype(np.uint8))
            
            trans_array = np.array(trans_array)
            np.savez_compressed(os.path.join(bin_dir, 't.npz'), t=trans_array)

            bin_zip_name = bin_dir.split('/')[-1]
            bin_zip_path = os.path.join(exp_dir, f'{bin_zip_name}.zip')
            os.system(f'zip -j {bin_zip_path} {bin_dir}/*')

            zip_file_size = os.path.getsize(bin_zip_path)

            print('final sum:', zip_file_size , 'B')
            print('final sum:', zip_file_size / 1024, 'KB')
            print('final sum:', zip_file_size / 1024 / 1024, 'MB')

    def init_qas(self, lseg):
        lc = self._xyz.shape[0]
        n_qs = 0
        rf_base = 7
        
        if lseg > lc:
            print('lseg > lc, set lseg == lc.')
            lseg = lc
        
        '''cal number of qas for opa, rot/euler, and SH0'''
        if (lc - 1) % lseg == 0:
            n_qs += (((lc - 1) // lseg)) * rf_base
        else:
            n_qs += (((lc - 1) // lseg) + 1) * rf_base
        
        '''cal number of qas for scale'''
        if lc % lseg == 0:
            n_qs += (lc // lseg) * 3
        else:
            n_qs += ((lc // lseg) + 1) * 3
        # n_qs = ((lc-1) // lseg + 1) * 7 + (lc // lseg + 1)*3
        
        self.qas = nn.ModuleList([])
        for i in range(n_qs): 
            self.qas.append(VanillaQuan(bit=8).cuda())
        self.lseg = lseg
        print('Init qa, length:', n_qs)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity)*0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, og_number_points=-1, spatial_lr_scale=-1):
        self.spatial_lr_scale = spatial_lr_scale
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        self.og_number_points = xyz.shape[0]

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]


        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        
        self.n_sh = (self.max_sh_degree + 1) ** 2

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))        
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree 
    
        
    def load_ply_cov(self, path, og_number_points=-1):
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        cov_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("cov_")]
        covs = np.zeros((xyz.shape[0], len(cov_names)))
        for idx, attr_name in enumerate(cov_names):
            covs[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._cov = nn.Parameter(torch.tensor(covs, dtype=torch.float, device="cuda").requires_grad_(False))
        self.active_sh_degree = self.max_sh_degree


    def load_ft_rots(self, path, og_number_points=-1):
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        self.og_number_points = xyz.shape[0]
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree 
        self.n_sh = (self.max_sh_degree + 1) ** 2

    def octree_coding(self, imp, merge_type, raht=False):
        features = torch.concat([
            self._opacity.detach(), 
            self._features_dc.detach().flatten(-2).contiguous(), 
            self._features_rest.detach().flatten(-2).contiguous(), 
            self._scaling.detach(), 
            self._rotation.detach()], -1).cpu().numpy()

        V, features, oct, paramarr, _, _ = create_octree_overall(
            self._xyz.detach().cpu().numpy(), 
            features,
            imp,
            depth=self.depth,
            oct_merge=merge_type)
        dxyz, _ = decode_oct(paramarr, oct, self.depth)
        
        if raht:
            # morton sort
            w, val, reorder = copyAsort(V)
            self.reorder = reorder
            self.res = haar3D_param(self.depth, w, val)
            self.res_inv = inv_haar3D_param(V, self.depth)
            self.scale_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        
        opacities = features[:, :1]
        features_dc = features[:, 1:4].reshape(-1, 1, 3)
        features_extra = features[:, 4:4 + 3 * (self.n_sh-1)].reshape(-1, self.n_sh - 1, 3)
        scales=features[:,49:52]
        rots=features[:,52:56]
        
        self.oct = oct
        self.oct_param = paramarr
        self._xyz = nn.Parameter(torch.tensor(dxyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
    
    def load_npz(self, exp_dir):
        if exp_dir[-4:] == '.zip':
            print('Assume the input file path is a .zip file')
            bin_dir = os.path.join(exp_dir[:-8], 'bins_tmp')
            if os.path.exists(bin_dir):
                shutil.rmtree(bin_dir)
            os.system(f'unzip {exp_dir} -d {bin_dir}')
        else:
            print('Assume the input file path is a dir that contains a \'bins\' dir')
            bin_dir = os.path.join(exp_dir, 'bins')
        
        print('load ply from:', bin_dir)
        trans_array = np.load(os.path.join(bin_dir, 't.npz'))["t"]
        
        depth = int(trans_array[0])
        self.depth = depth
        
        oct_vals = np.load(os.path.join(bin_dir , 'oct.npz'))
        
        octree = oct_vals["points"]
        oct_param = oct_vals["params"]
        self.og_number_points = octree.shape[0]

        dxyz, V = decode_oct(oct_param, octree, depth)

        self._xyz = nn.Parameter(torch.tensor(dxyz, dtype=torch.float, device="cuda").requires_grad_(False))
        n_points = dxyz.shape[0]
        
        ntk = np.load(os.path.join(bin_dir , 'ntk.npz'))["ntk"]
        cb = torch.tensor(np.load(os.path.join(bin_dir , 'um.npz'))["umap"])
        # print('ntk.shape', ntk.shape)
        # print('cb.shape', cb.shape)
        
        features_rest = torch.zeros([ntk.shape[0], cb.shape[1]])
        for i in range(ntk.shape[0]):
            features_rest[i] = cb[int(ntk[i])]
        self.n_sh = (self.max_sh_degree + 1) ** 2
        self._features_rest = nn.Parameter(features_rest.to("cuda")).contiguous().reshape(-1, self.n_sh - 1, 3).requires_grad_(False)
        
        # self._features_rest = nn.Parameter(
        #     torch.matmul(
        #         F.one_hot(torch.tensor(ntk, dtype=torch.long, device="cuda")).float(), 
        #         torch.tensor(cb, dtype=torch.float, device="cuda")
        #     ).contiguous().reshape(-1, self.n_sh - 1, 3).requires_grad_(False))
        
        # print('gaussian model, line 1027, trans_array', trans_array.shape, trans_array)
        
        oef_vals = np.load(os.path.join(bin_dir,'orgb.npz'))
        orgb_f = torch.tensor(oef_vals["f"], dtype=torch.float, device="cuda")
        q_orgb_i = torch.tensor(oef_vals["i"].astype(np.float32), dtype=torch.float, device="cuda").reshape(7, -1).contiguous().transpose(0, 1)
        q_scale_i = torch.tensor(np.load(os.path.join(bin_dir, 'ct.npz'))["i"], dtype=torch.float, device="cuda").reshape(3, -1).contiguous().transpose(0, 1)
        # q_orgb_i = torch.tensor(oef_vals["i"].astype(np.float32), dtype=torch.float, device="cuda").reshape(-1, 7)
        # q_scale_i = torch.tensor(np.load(os.path.join(bin_dir, 'ct.npz'))["i"], dtype=torch.float, device="cuda").reshape(-1, 3)
        
        # print('rf_orgb_f size', orgb_f.shape)
        # print('q_rf_orgb_i.shape', q_orgb_i.shape)
        # print('q_scale_i.shape', q_scale_i.shape)
        
        
        lseg = int(trans_array[1])
        self.lseg = lseg
        
        '''dequant'''
        qa_cnt = 2
        rf_orgb = []
        rf_len = q_orgb_i.shape[0]
        assert rf_len + 1 == n_points
        if rf_len % self.lseg == 0:
            n_rf = rf_len // self.lseg
        else:
            n_rf = rf_len // self.lseg + 1
        for i in range(7):
            rf_i = torch_vanilla_dequant(q_orgb_i[:, i], lseg, trans_array[qa_cnt:qa_cnt+2*n_rf])
            rf_orgb.append(rf_i.reshape(-1, 1))
            qa_cnt += 2*n_rf
        rf_orgb = torch.concat(rf_orgb, axis=-1)
        
        
        de_scale = []
        scale_len = q_scale_i.shape[0]
        assert scale_len == n_points
        if scale_len % self.lseg == 0:
            n_scale = scale_len // self.lseg
        else:
            n_scale = scale_len // self.lseg + 1
        for i in range(3):
            scale_i = torch_vanilla_dequant(q_scale_i[:, i], lseg, trans_array[qa_cnt:qa_cnt+2*n_scale])
            de_scale.append(scale_i.reshape(-1, 1))
            qa_cnt += 2*n_scale
        de_scale = torch.concat(de_scale, axis=-1).to("cuda")
        self._scaling = nn.Parameter(de_scale.requires_grad_(False))
        
        # print('qa_cnt', qa_cnt, 'trans', trans_array.shape)
        # print('rf_orgb.shape, de_scale.shape', rf_orgb.shape, de_scale.shape)
        
        C = torch.concat([orgb_f.reshape(1, -1), rf_orgb], 0)
        
        w, val, reorder = copyAsort(V)

        self.reorder = reorder  
        res_inv = inv_haar3D_param(V, depth)
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

        CT_yuv_q_temp = C[pos.astype(int)]
        raht_features = torch.zeros(C.shape).cuda()
        OC = torch.zeros(C.shape).cuda()

        for i in range(depth*3):
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

        raht_features[reorder] = OC
        
        self._opacity = nn.Parameter(raht_features[:, :1].requires_grad_(False))
        self._euler = nn.Parameter(raht_features[:, 1:4].requires_grad_(False))
        self._features_dc = nn.Parameter(raht_features[:, 4:].unsqueeze(1).requires_grad_(False))
        
        self.active_sh_degree = self.max_sh_degree
        
        if exp_dir[-4:] == '.zip':
            os.system(f'rm -rf {bin_dir}')
            

    def vq_fe(self, imp, codebook_size, batch_size, steps):
        features_extra = self._features_rest.detach().flatten(-2)
        codebook, vq_indices = vq_features(
            features_extra,
            imp,
            codebook_size,
            batch_size,
            steps,
        )

        self._feature_indices = nn.Parameter(vq_indices.detach().contiguous(), requires_grad=False)
        self._features_rest = nn.Parameter(codebook.detach().contiguous(), requires_grad=True)
        
        

    def load_ply_euler(self, path, og_number_points=-1):
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        print('opacities save shape', opacities.shape)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        print('scales save shape', scales.shape)

        euler_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("euler_")]
        eulers = np.zeros((xyz.shape[0], len(euler_names)))
        for idx, attr_name in enumerate(euler_names):
            eulers[:, idx] = np.asarray(plydata.elements[0][attr_name])
        print('eulers save shape', eulers.shape)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._euler = nn.Parameter(torch.tensor(eulers, dtype=torch.float, device="cuda").requires_grad_(False))
        self.active_sh_degree = self.max_sh_degree
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
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
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # print('prune unfound:', group["name"])
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def prune_gaussians(self, mask):
        valid_points_mask = ~mask
        self._xyz = self._xyz[valid_points_mask]
        self._features_dc = self._features_dc[valid_points_mask]
        self._features_rest = self._features_rest[valid_points_mask]
        self._opacity = self._opacity[valid_points_mask]
        self._scaling = self._scaling[valid_points_mask]
        self._rotation = self._rotation[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
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
                # print('cat unfound:', group["name"])
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    