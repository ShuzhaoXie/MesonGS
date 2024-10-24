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
from argparse import ArgumentParser
from os import makedirs
from time import gmtime, strftime

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args_render
from gaussian_renderer import GaussianModel, render
from lpipsPyTorch import lpips
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.system_utils import MAIN_DIR
from utils.render_utils import generate_path, create_color_videos

def render_video(model_path, name, scene, pipeline, gaussians, background, render_args):
    n_fames = 480
    cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
    makedirs(os.path.join(model_path, name, render_args.save_dir_name), exist_ok=True)
    render_path = os.path.join(model_path, name, render_args.save_dir_name, "renders")    
    makedirs(render_path, exist_ok=True)
    for idx, cam in enumerate(tqdm(cam_traj, desc="Rendering progress")):
        render_results = render(cam, gaussians, pipeline, background, debug=render_args.debug, clamp_color=True)
        rendering = render_results["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    
    create_color_videos(base_dir=render_args.video_dir,
            input_dir=render_path, 
            out_name='test1', 
            num_frames=n_fames)
    

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, render_args):
    makedirs(os.path.join(model_path, name, render_args.save_dir_name), exist_ok=True)
    render_path = os.path.join(model_path, name, render_args.save_dir_name, "renders")    
    gts_path = os.path.join(model_path, name, render_args.save_dir_name, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    ssims = []
    psnrs = []
    lpipss = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_results = render(view, gaussians, pipeline, background, debug=render_args.debug, clamp_color=True)
        rendering = render_results["render"]
        gt = view.original_image[0:3, :, :]
        # print('rendering.shape', rendering.shape)
        # print('gt.shape', gt.shape)
        # images = torch.cat((images, rendering.unsqueeze(0)), dim=0)
        # gts = torch.cat((gts, gt.unsqueeze(0)), dim=0)
        if not render_args.quick:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        psnrs.append(psnr(rendering, gt))
        if not render_args.quick:
            ssims.append(ssim(rendering, gt))
            # print(psnrs)
            lpipss.append(lpips(rendering, gt, net_type='vgg'))

    ssim_val = torch.tensor(ssims).mean()
    psnr_val = torch.stack(psnrs).mean()
    lpips_val = torch.tensor(lpipss).mean()
    
    print('psnr, ssim, lpips')
    print("{:>12.7f}".format(psnr_val, ".5")+ '\t' + "{:>12.7f}".format(ssim_val, ".5")+ '\t' + "{:>12.7f}".format(lpips_val, ".5"))
    print("")
    return psnr_val.item(), ssim_val.item(), lpips_val.item()



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, render_args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, 
                      shuffle=False,
                      given_ply_path=render_args.given_ply_path, 
                      dec_cov=render_args.dec_cov, 
                      dec_euler=render_args.dec_euler,
                      dec_npz=render_args.dec_npz)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        # print('bg_color', bg_color)
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        metric_vals = []
        
        if render_args.render_video:
            render_video(dataset.model_path, 'train', scene, pipeline, gaussians, background, render_args)
            
        if not render_args.skip_train:
            psnrv, ssimv, lpipsv = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_args)
            metric_vals.extend([psnrv, ssimv, lpipsv])
        else:
            metric_vals.extend([0, 0, 0])

        if not render_args.skip_test:
            psnrv, ssimv, lpipsv = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, render_args)
            metric_vals.extend([psnrv, ssimv, lpipsv])
        else:
            metric_vals.extend([0, 0, 0])
        
        return metric_vals

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--all_bits", action="store_true")
    # parser.add_argument("--num_bits", default=32, type=int)
    parser.add_argument("--given_ply_path", default='', type=str)
    parser.add_argument("--save_dir_name", default='', type=str)
    parser.add_argument("--scene_name", default='', type=str)                                                                    
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--save_imp", action="store_true")
    parser.add_argument("--dec_cov", action="store_true")
    parser.add_argument("--dec_euler", action="store_true")
    parser.add_argument("--dec_npz", action="store_true")
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument("--log_name", default='', type=str)
    parser.add_argument("--video_dir", default='', type=str)
    parser.add_argument("--quick", action="store_true")
    # args = get_combined_args(parser)
    args = get_combined_args_render(parser)
    # args = parser.parse_args(sys.argv[1:])
    print(args)
    render_args = args
    # render_args = parser.parse_args()
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    csv_path = os.path.join(f'{MAIN_DIR}/exp_data/csv', args.log_name + '.csv')
    
    if not os.path.exists(f'{MAIN_DIR}/exp_data/csv'):
        makedirs(f'{MAIN_DIR}/exp_data/csv')

    if not os.path.exists(csv_path):
        f = open(csv_path, 'a+')
        wtr = csv.writer(f)
        wtr.writerow(['name', 'train_p', 'train_s', 'train_l', 'test_p', 'test_s', 'test_l'])
        f.close()
    
    if not os.path.exists(f'{MAIN_DIR}/exp_data/txt'):
        makedirs(f'{MAIN_DIR}/exp_data/txt')
    
    print(model.extract(args))
    metric_vals = render_sets(
        model.extract(args), 
        args.iteration, 
        pipeline.extract(args), 
        render_args,) 
    
    # if args.render_video:
    #     render_video(
    #         model.extract(args),
            
    #     )
    source_path = model.extract(args).source_path
    source_name = source_path.split('/')[-1]
    cur_time = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    row = []
    row.append(args.scene_name + '_' + args.save_dir_name)
    f = open(csv_path, 'a+')
    wtr = csv.writer(f)
    row.extend(metric_vals)
    wtr.writerow(row)
    f.close()
