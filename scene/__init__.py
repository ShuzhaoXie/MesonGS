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

import json
import os
import random

from arguments import ModelParams
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from utils.system_utils import searchForMaxIteration


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, 
                 load_iteration=None, shuffle=True, 
                 resolution_scales=[1.0], given_ply_path=None, dec_cov=False, 
                 dec_euler=False, dec_npz=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        
        if given_ply_path == '' or given_ply_path == None:
            if load_iteration:
                if load_iteration == -1 :
                    self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
                else:
                    self.loaded_iter = load_iteration
                print("Loading trained model at iteration {}".format(self.loaded_iter))
                given_ply_path = os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply")
        else:
            self.loaded_iter = -1
        
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras", len(scene_info.train_cameras))
            # print(len(scene_info.train_cameras))
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras", len(scene_info.test_cameras))
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        
        if given_ply_path:
            if dec_cov:
                self.gaussians.load_ply_cov(given_ply_path,
                                                og_number_points=-1)
                print("Load decoded ply from", given_ply_path)
            elif dec_euler:
                self.gaussians.load_ply_euler(given_ply_path,
                                                og_number_points=-1)
                print("Load decoded euler ply from", given_ply_path)
            elif dec_npz:
                self.gaussians.load_npz(given_ply_path)
                print("Load decoded npz from", given_ply_path)
            else:
                self.gaussians.load_ply(given_ply_path,
                                                og_number_points=len(scene_info.point_cloud.points),
                                                spatial_lr_scale=self.cameras_extent)
                print("Load ply from", given_ply_path)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
        return point_cloud_path

    def save_ft(self, iteration):
        pc_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        print('save_npz gaussians:', pc_path)
        self.gaussians.save_npz(os.path.join(pc_path, "pc_npz"))
        return os.path.getsize(os.path.join(pc_path, "pc_npz", "bins.zip"))
        
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]