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

import csv
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import ft_render, render
from lpipsPyTorch import lpips
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def cal_sens(
        gaussians,
        views,
        pipeline, 
        background,
    ):
    scaling = gaussians.get_scaling.detach()
    cov3d = gaussians.covariance_activation(
        torch.tensor(scaling, device='cuda'), 1.0, torch.tensor(gaussians.get_rotation.detach(), device='cuda')
    ).requires_grad_(True)
    h1 = gaussians._features_dc.register_hook(lambda grad: grad.abs())
    h2 = gaussians._features_rest.register_hook(lambda grad: grad.abs())
    h3 = cov3d.register_hook(lambda grad: grad.abs())
    gaussians._features_dc.grad = None
    gaussians._features_rest.grad = None
    num_pixels = 0

    for view in tqdm(views, desc="Calculating c3dgs importance"):
        rendering = render(
            view,
            gaussians,
            pipeline,
            background,
            cov3d=cov3d,
            debug=False,
            clamp_color=False,
            meson_count=False,
            f_count=False
        )["render"]
        loss = rendering.sum()
        loss.backward()
        num_pixels += rendering.shape[1]*rendering.shape[2]
    importance = torch.cat(
        [gaussians._features_dc.grad, gaussians._features_rest.grad],
        1,
    ).flatten(-2)/num_pixels
    cov_grad = cov3d.grad/num_pixels
    h1.remove()
    h2.remove()
    h3.remove()
    torch.cuda.empty_cache()
    color_imp = importance.detach()
    cov_imp = cov_grad.detach()
    color_imp_n = color_imp.amax(-1)
    cov_imp_n = cov_imp.amax(-1)
    return color_imp_n * torch.pow(cov_imp_n, args.cov_beta)

def pre_volume(volume, beta):
    # volume = torch.tensor(volume)
    index = int(volume.shape[0] * 0.9)
    sorted_volume, _ = torch.sort(volume, descending=True)
    kth_percent_largest = sorted_volume[index]
    # Calculate v_list
    v_list = torch.pow(volume / kth_percent_largest, beta)
    return v_list

def cal_imp(
        gaussians,
        views,
        pipeline, 
        background
    ):
    beta_list = {
        'chair': 0.03,
        'drums': 0.05,
        'ficus': 0.03,
        'hotdog': 0.03,
        'lego': 0.05,
        'materials': 0.03,
        'mic': 0.03,
        'ship': 0.03,
        'bicycle': 0.03,
        'bonsai': 0.1,
        'counter': 0.1,
        'garden': 0.1,
        'kitchen': 0.1,
        'room': 0.1,
        'stump': 0.01,
    }   
    
    full_opa_imp = None 

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_results = render(view, gaussians, pipeline, background, debug=True, clamp_color=True, meson_count=True, f_count=False, depth_count=False)
        if full_opa_imp is not None:
            full_opa_imp += render_results["imp"]
        else:
            full_opa_imp = render_results["imp"]
            
        del render_results
        
    volume = torch.prod(gaussians.get_scaling, dim=1)

    v_list = pre_volume(volume, beta_list.get(pipeline.scene_imp, 0.1))
    imp = v_list * full_opa_imp
    
    return imp.detach()
    
def prune_mask(percent, imp):
    sorted_tensor, _ = torch.sort(imp, dim=0)
    index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
    value_nth_percentile = sorted_tensor[index_nth_percentile]
    prune_mask = (imp <= value_nth_percentile).squeeze()
    return prune_mask

lr_scale_list = {
    'chair': 0.4,
    'drums': 0.4,
    'ficus': 0.4,
    'hotdog': 0.1,
    'lego': 0.4,
    'materials': 0.1,
    'mic': 0.1,
    'ship': 0.1,
    'bicycle': 0.15,
    'bonsai': 0.1,
    'counter': 0.1,
    'garden': 0.1,
    'kitchen': 0.1,
    'room': 0.1,
    'stump': 0.01,
    'flowers': 0.05,
    'treehill': 0.05,
    'drjohnson': 0.1,
    'playroom': 0.1,
    'train': 0.05,
    'truck': 0.2
}


universal_config = {
    # 'lseg': {
    #     'chair': 4000,
    #     'drums': 4000,
    #     'ficus': 4000,
    #     'hotdog': 2000,
    #     'lego': 2000,
    #     'materials': 200,
    #     'mic': 4000,
    #     'ship': 2000,
    #     'bicycle': 20000,
    #     'bonsai': 20000,
    #     'counter': 5000,
    #     'garden': 20000,
    #     'kitchen': 1000,
    #     'room': 20000,
    #     'stump': 20000,
    #     'drjohnson': 20000,
    #     'playroom': 20000,
    #     'truck': 25000,
    #     'train': 3000
    # },
    'n_block': {
        'bicycle' : 66,
        'bonsai' : 66,
        'counter' : 66,
        'garden' : 66,
        'kitchen' : 66,
        'room': 66,
        'stump': 66,
        'flowers': 66,
        'treehill': 66,
        'drjohnson': 66,
        'playroom': 66,
        'train': 66,
        'truck': 66,
        'chair': 62,
        'drums': 66,
        'ficus': 50,
        'hotdog': 66,
        'lego': 66,
        'materials': 66,
        'mic': 52,
        'ship': 66
    },
    'cb': {
        'chair': 2048,
        'drums': 2048,
        'ficus': 2048,
        'hotdog': 4096,
        'lego': 4096,
        'materials': 4096,
        'mic': 2048,
        'ship': 8192,
        'bicycle': 2048,
        'bonsai': 2048,
        'counter': 2048,
        'garden': 2048,
        'kitchen': 8192,
        'room': 2048,
        'stump': 2048,
        'flowers': 2048,
        'treehill': 2048,
        'drjohnson': 2048,
        'playroom': 2048,
        'train': 4096,
        'truck': 4096
    },
    'depth': {
        'chair': 14,
        'drums': 14,
        'ficus': 14,
        'hotdog': 14,
        'lego': 14,
        'materials': 14,
        'mic': 14,
        'ship': 14,
        'bicycle': 20,
        'bonsai': 19,
        'counter': 19,
        'garden': 20,
        'kitchen': 19,
        'room': 19,
        'stump': 20,
        'flowers': 20,
        'treehill': 20,
        'drjohnson': 20,
        'playroom': 20,
        'train': 20,
        'truck': 20
    },
    'prune':  {
        'chair': 0.06,
        'drums': 0.18,
        'ficus': 0.28,
        'hotdog':  0.32,
        'lego': 0.1,
        'materials': 0.1,
        'mic': 0.3,
        'ship': 0.18,
        'bicycle': 0.3,
        'bonsai': 0.24,
        'counter': 0.12,
        'garden': 0.18,
        'kitchen': 0.14,
        'room': 0.22,
        'stump': 0.2,
        'flowers': 0.2,
        'treehill': 0.2,
        'drjohnson': 0.41,
        'playroom': 0.0,
        'train': 0.12,
        'truck': 0.38
    }
}

config2 = {
    # 'lseg': {
    #     'chair': 4000,
    #     'drums': 4000,
    #     'ficus': 4000,
    #     'hotdog': 2000,
    #     'lego': 2000,
    #     'materials': 200,
    #     'mic': 4000,
    #     'ship': 2000,
    #     'bicycle': 20000,
    #     'bonsai': 20000,
    #     'counter': 5000,
    #     'garden': 20000,
    #     'kitchen': 1000,
    #     'room': 20000,
    #     'stump': 20000,
    #     'drjohnson': 20000,
    #     'playroom': 20000,
    #     'truck': 25000,
    #     'train': 3000
    # },
    'n_block': {
        'bicycle' : 50,
        'bonsai' : 50,
        'counter' : 50,
        'garden' : 50,
        'kitchen' : 50,
        'room': 50,
        'stump': 50,
        'flowers': 50,
        'treehill': 50,
        'drjohnson': 50,
        'playroom': 50,
        'train': 50,
        'truck': 50,
        'chair': 48,
        'drums': 50,
        'ficus': 50,
        'hotdog': 50,
        'lego': 50,
        'materials': 50,
        'mic': 42,
        'ship': 50
    },
    'cb': {
        'chair': 2048,
        'drums': 2048,
        'ficus': 2048,
        'hotdog': 4096,
        'lego': 4096,
        'materials': 4096,
        'mic': 2048,
        'ship': 8192,
        'bicycle': 2048,
        'bonsai': 2048,
        'counter': 2048,
        'garden': 2048,
        'kitchen': 8192,
        'room': 2048,
        'stump': 2048,
        'flowers': 2048,
        'treehill': 2048,
        'drjohnson': 2048,
        'playroom': 2048,
        'train': 4096,
        'truck': 4096
    },
    'depth': {
        'chair': 14,
        'drums': 14,
        'ficus': 14,
        'hotdog': 14,
        'lego': 14,
        'materials': 14,
        'mic': 14,
        'ship': 14,
        'bicycle': 20,
        'bonsai': 19,
        'counter': 19,
        'garden': 20,
        'kitchen': 19,
        'room': 19,
        'stump': 20,
        'flowers': 20,
        'treehill': 20,
        'drjohnson': 20,
        'playroom': 20,
        'train': 20,
        'truck': 20
    },
    'prune':  {
        'chair': 0.06,
        'drums': 0.18,
        'ficus': 0.28,
        'hotdog':  0.32,
        'lego': 0.1,
        'materials': 0.1,
        'mic': 0.3,
        'ship': 0.18,
        'bicycle': 0.3,
        'bonsai': 0.24,
        'counter': 0.12,
        'garden': 0.18,
        'kitchen': 0.14,
        'room': 0.22,
        'stump': 0.2,
        'flowers': 0.45,
        'treehill': 0.45,
        'drjohnson': 0.41,
        'playroom': 0.0,
        'train': 0.12,
        'truck': 0.38
    }
}


# config3 = {
#     'n_block': {
#         'bicycle' : 57,
#         'bonsai' : 57,
#         'counter' : 57,
#         'garden' : 57,
#         'kitchen' : 57,
#         'room': 57,
#         'stump': 57,
#         'flowers': 57,
#         'treehill': 57,
#         'drjohnson': 57,
#         'playroom': 57,
#         'train': 57,
#         'truck': 57,
#         'chair': 52,
#         'drums': 57,
#         'ficus': 57,
#         'hotdog': 57,
#         'lego': 57,
#         'materials': 57,
#         'mic': 48,
#         'ship': 57
#     },
#     'cb': {
#         'chair': 2048,
#         'drums': 2048,
#         'ficus': 2048,
#         'hotdog': 4096,
#         'lego': 4096,
#         'materials': 4096,
#         'mic': 2048,
#         'ship': 8192,
#         'bicycle': 2048,
#         'bonsai': 2048,
#         'counter': 2048,
#         'garden': 2048,
#         'kitchen': 8192,
#         'room': 2048,
#         'stump': 2048,
#         'flowers': 2048,
#         'treehill': 2048,
#         'drjohnson': 2048,
#         'playroom': 2048,
#         'train': 4096,
#         'truck': 4096
#     },
#     'depth': {
#         'chair': 14,
#         'drums': 14,
#         'ficus': 14,
#         'hotdog': 14,
#         'lego': 14,
#         'materials': 14,
#         'mic': 14,
#         'ship': 14,
#         'bicycle': 20,
#         'bonsai': 19,
#         'counter': 19,
#         'garden': 20,
#         'kitchen': 19,
#         'room': 19,
#         'stump': 20,
#         'flowers': 20,
#         'treehill': 20,
#         'drjohnson': 20,
#         'playroom': 20,
#         'train': 20,
#         'truck': 20
#     },
#     'prune':  {
#         'chair': 0.16,
#         'drums': 0.28,
#         'ficus': 0.38,
#         'hotdog':  0.42,
#         'lego': 0.2,
#         'materials': 0.2,
#         'mic': 0.4,
#         'ship': 0.28,
#         'bicycle': 0.4,
#         'bonsai': 0.34,
#         'counter': 0.22,
#         'garden': 0.28,
#         'kitchen': 0.24,
#         'room': 0.32,
#         'stump': 0.3,
#         'flowers': 0.5,
#         'treehill': 0.5,
#         'drjohnson': 0.41,
#         'playroom': 0.2,
#         'train': 0.22,
#         'truck': 0.4
#     }
# }

config3 = {
    'n_block': {
        'bicycle' : 57,
        'bonsai' : 57,
        'counter' : 57,
        'garden' : 57,
        'kitchen' : 57,
        'room': 57,
        'stump': 57,
        'flowers': 57,
        'treehill': 57,
        'drjohnson': 57,
        'playroom': 57,
        'train': 57,
        'truck': 57,
        'chair': 52,
        'drums': 57,
        'ficus': 57,
        'hotdog': 57,
        'lego': 57,
        'materials': 57,
        'mic': 48,
        'ship': 57
    },
    'cb': {
        'chair': 2048,
        'drums': 2048,
        'ficus': 2048,
        'hotdog': 4096,
        'lego': 4096,
        'materials': 4096,
        'mic': 2048,
        'ship': 8192,
        'bicycle': 2048,
        'bonsai': 2048,
        'counter': 2048,
        'garden': 2048,
        'kitchen': 8192,
        'room': 2048,
        'stump': 2048,
        'flowers': 2048,
        'treehill': 2048,
        'drjohnson': 2048,
        'playroom': 2048,
        'train': 4096,
        'truck': 4096
    },
    'depth': {
        'chair': 14,
        'drums': 14,
        'ficus': 14,
        'hotdog': 14,
        'lego': 14,
        'materials': 14,
        'mic': 14,
        'ship': 14,
        'bicycle': 20,
        'bonsai': 19,
        'counter': 19,
        'garden': 20,
        'kitchen': 19,
        'room': 19,
        'stump': 20,
        'flowers': 20,
        'treehill': 20,
        'drjohnson': 20,
        'playroom': 20,
        'train': 20,
        'truck': 20
    },
    'prune':  {
        'chair': 0.16,
        'drums': 0.28,
        'ficus': 0.38,
        'hotdog':  0.42,
        'lego': 0.2,
        'materials': 0.2,
        'mic': 0.4,
        'ship': 0.28,
        'bicycle': 0.4,
        'bonsai': 0.34,
        'counter': 0.22,
        'garden': 0.28,
        'kitchen': 0.24,
        'room': 0.32,
        'stump': 0.3,
        'flowers': 0.5,
        'treehill': 0.5,
        'drjohnson': 0.41,
        'playroom': 0.2,
        'train': 0.22,
        'truck': 0.4
    }
}


# 'prune':  {
#         'chair': 0.06,
#         'drums': 0.18,
#         'ficus': 0.28,
#         'hotdog':  0.32,
#         'lego': 0.1,
#         'materials': 0.1,
#         'mic': 0.3,
#         'ship': 0.18,
#         'bicycle': 0.3,
#         'bonsai': 0.24,
#         'counter': 0.12,
#         'garden': 0.18,
#         'kitchen': 0.14,
#         'room': 0.22,
#         'stump': 0.2,
#         'drjohnson': 0.41,
#         'playroom': 0.0,
#         'train': 0.12,
#         'truck': 0.38
#     }


cb_const = 256
prune_const = 0.66
nerf_syn_small_config = {
    'lseg': {
        'chair': 1000000,
        'drums': 1000000,
        'ficus': 1000000,
        'hotdog': 1000000,
        'lego': 1000000,
        'materials': 1000000,
        'mic': 1000000,
        'ship': 1000000
    },
    'cb': {
        'chair': cb_const,
        'drums': cb_const,
        'ficus': cb_const,
        'hotdog': cb_const,
        'lego': cb_const,
        'materials': cb_const,
        'mic': cb_const,
        'ship': cb_const
    },
    'depth': {
        'chair': 12,
        'drums': 12,
        'ficus': 12,
        'hotdog': 12,
        'lego': 12,
        'materials': 12,
        'mic': 12,
        'ship': 12
    },
    'prune': {
        'chair': prune_const,
        'drums': prune_const,
        'ficus': prune_const,
        'hotdog': prune_const,
        'lego': prune_const,
        'materials': prune_const,
        'mic': prune_const,
        'ship': prune_const
    }
}

def evaluate_test(scene, dataset, pipe, background, iteration=0):
    model_path = dataset.model_path
    cams = scene.getTestCameras()

    ssims = []
    lpipss = []
    psnrs = []

    render_path = os.path.join(model_path, 'test', f'iter_{iteration}')
    os.makedirs(render_path, exist_ok=True)
    
    gts_path = os.path.join(model_path, 'gt', f'iter_{iteration}')
    os.makedirs(gts_path, exist_ok=True)

    for idx, viewpoint in enumerate(tqdm(cams)):
        image = ft_render(
            viewpoint, 
            scene.gaussians, 
            pipe, 
            background,
            training=False,
            raht=dataset.raht,
            debug=dataset.debug,
            per_channel_quant=dataset.per_channel_quant,
            per_block_quant=dataset.per_block_quant,
            clamp_color=True)["render"]
        gt_image = viewpoint.original_image[0:3, :, :].to("cuda")
        # print(gt_image.max(), gt_image.min())
        torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        psnrs.append(psnr(image.unsqueeze(0), gt_image).unsqueeze(0))
        
        ssims.append(ssim(image, gt_image))
        lpipss.append(lpips(image, gt_image, net_type='vgg'))

    
    psnr_val = torch.tensor(psnrs).mean()
    ssim_val = torch.tensor(ssims).mean()
    lpips_val = torch.tensor(lpipss).mean()
    
    return psnr_val.item(), ssim_val.item(), lpips_val.item()


def training(dataset, opt, pipe, testing_iterations, given_ply_path=None):
    # print('dataset.eval', dataset.eval)
    print('debug', dataset.debug)

    # magnify the lr scale of the covergence process is too slow.
    opt.finetune_lr_scale = lr_scale_list[pipe.scene_imp] * 4 
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, depth=dataset.depth, num_bits=dataset.num_bits)
    scene = Scene(dataset, gaussians, given_ply_path=given_ply_path)
    
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # nspd_mask = gaussians.check_spd()
    # print('nspd_mask.shape', nspd_mask.shape)
    
    with torch.no_grad():
        imp = cal_imp(gaussians, scene.getTrainCameras(), pipe, background)
    # imp = cal_sens(gaussians, scene.getTrainCameras(), pipe, background)

    pmask = prune_mask(dataset.percent, imp)
    imp = imp[torch.logical_not(pmask)]

    gaussians.prune_points(pmask)
    
    gaussians.octree_coding(
        imp,
        dataset.oct_merge,
        raht=dataset.raht
    )
    
    if dataset.per_block_quant:
        gaussians.init_qas(dataset.n_block)

    gaussians.vq_fe(imp, dataset.codebook_size, dataset.batch_size, dataset.steps)
        
    print('Test Model (offline)...')
    with torch.no_grad():
        psnr_val, ssim_val, lpips_val = evaluate_test(
            scene,
            dataset,
            pipe,
            background,
            iteration=0
        )
        zip_size = scene.save_ft("0", pipe, per_channel_quant=dataset.per_channel_quant, per_block_quant=dataset.per_block_quant)
        zip_size = zip_size / 1024 / 1024 # to MB
        row = []
        row.append(pipe.scene_imp)
        row.extend([0, psnr_val, ssim_val, lpips_val, zip_size])
        f = open(dataset.csv_path, 'a+')
        wtr = csv.writer(f)
        wtr.writerow(row)
        f.close()
        print("Testset Evaluating {}. PSNR: {}, SSIM: {}, LIPIS: {}".format(0, psnr_val, ssim_val, lpips_val, zip_size))
    
    gaussians.finetuning_setup(opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
        
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    psnr_train = 0
    for iteration in range(1, opt.iterations + 1):    
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        # try:
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        
        render_pkg = ft_render(
            viewpoint_cam, 
            gaussians, 
            pipe, 
            background,
            training=True,
            raht=dataset.raht,
            debug=dataset.debug,
            per_channel_quant=dataset.per_channel_quant,
            per_block_quant=dataset.per_block_quant,
            clamp_color=dataset.clamp_color)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # Log and save
            cur_psnr, _, _ = training_report(
                tb_writer, 
                iteration, 
                Ll1, 
                loss, 
                l1_loss, 
                iter_start.elapsed_time(iter_end), 
                testing_iterations, 
                scene, 
                ft_render, 
                pipe.scene_imp, 
                dataset.csv_path, 
                dataset.model_path, 
                (pipe, background, False, 1.0, None, 
                 dataset.raht, dataset.per_channel_quant, 
                 dataset.per_block_quant, False, True))
            
            if cur_psnr > psnr_train:
                psnr_train = cur_psnr
                print("\n Saving best Gaussians on Train Set.")
                scene.save_ft('best', pipe, per_channel_quant=dataset.per_channel_quant, per_block_quant=dataset.per_block_quant)
 
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.update_learning_rate(iteration+30000)
        # except Exception as e:
        #     print('error but go')

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
        tb_writer, 
        iteration, 
        Ll1, 
        loss, 
        l1_loss, 
        elapsed, 
        testing_iterations,
        scene : Scene, 
        renderFunc, 
        scene_name, 
        csv_path, 
        model_path, 
        renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    psnr_val, ssim_val, lpips_val = 0, 0, 0
    # Report test and samples of training set
    if iteration in testing_iterations:
        # print('trianing report, iteration', iteration)
        torch.cuda.empty_cache()
        config = {
            'train' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
            'test' : scene.getTestCameras()
        }
        
        for mode in ['train', 'test']:
            cams = config[mode]
            images = torch.tensor([], device="cuda")
            gts = torch.tensor([], device="cuda")
            
            if mode == 'test':
                ssims = []
                lpipss = []
                
            render_path = os.path.join(model_path, mode, f'iter_{iteration}')
            os.makedirs(render_path, exist_ok=True)
            
            for idx, viewpoint in enumerate(cams):
                if mode == 'test':
                    image = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"]
                    torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                    gt_image = viewpoint.original_image[0:3, :, :].to("cuda")
                    torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                    ssims.append(ssim(image, gt_image))
                    lpipss.append(lpips(image, gt_image, net_type='vgg'))
                if mode == 'train':
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                images = torch.cat((images, image.unsqueeze(0)), dim=0)
                gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)
            
            l1_test = l1_loss(images, gts)
            psnr_val = psnr(images, gts).mean()
            
            if mode == 'train':           
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, mode, l1_test, psnr_val))
                if tb_writer:
                    tb_writer.add_scalar(mode + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(mode + '/loss_viewpoint - psnr', psnr_val, iteration)
                torch.cuda.empty_cache()
            
            if mode == 'test':
                ssim_val = torch.tensor(ssims).mean()
                lpips_val = torch.tensor(lpipss).mean()
                
                torch.cuda.empty_cache()
            
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        
        zip_size = scene.save_ft(iteration, pipe, per_channel_quant=dataset.per_channel_quant, per_block_quant=dataset.per_block_quant)
        zip_size = zip_size / 1024 / 1024
        
        row = []
        row.append(scene_name)
        row.extend([iteration, psnr_val.item(), ssim_val.item(), lpips_val.item(), zip_size])
        f = open(csv_path, 'a+')
        wtr = csv.writer(f)
        wtr.writerow(row)
        f.close()
        print("Testset Evaluating {}. PSNR: {}, SSIM: {}, LIPIS: {}".format(iteration, psnr_val, ssim_val, lpips_val, zip_size))

    return psnr_val, ssim_val, lpips_val

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--given_ply_path", default='', type=str)
    args = parser.parse_args(sys.argv[1:])
    args.test_iterations = [int(x) for x in range(0, args.iterations+1, 400)]
    print('args.test_iterations', args.test_iterations)
    
    # args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    
    # use given config
    if pipe.hyper_config == 'universal':
        used_config = universal_config
    elif pipe.hyper_config == 'syn_small':
        used_config = nerf_syn_small_config
    elif pipe.hyper_config == 'config2':
        used_config = config2
    elif pipe.hyper_config == 'config3':
        used_config = config3
    else:
        used_config = None
    # use given config
    
    print('hyper_config', pipe.hyper_config)
    if used_config != None:
        dataset.percent = used_config['prune'][pipe.scene_imp]
        dataset.codebook_size = used_config['cb'][pipe.scene_imp]
        # dataset.lseg = used_config['lseg'][pipe.scene_imp]
        dataset.depth = used_config['depth'][pipe.scene_imp]
        dataset.n_block = used_config['n_block'][pipe.scene_imp]
        
    if not os.path.exists(dataset.csv_path):
        f = open(dataset.csv_path, 'a+')
        wtr = csv.writer(f)
        wtr.writerow(['name', 'iteration', 'psnr', 'ssim', 'lpips', 'size'])
        f.close()
        
    training(dataset, op.extract(args), pipe, args.test_iterations, args.given_ply_path)

    # All done
    print("\nTraining complete.")
