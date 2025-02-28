/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_imp", &RasterizeGaussiansImpCUDA);
  m.def("rasterize_gaussians_count", &RasterizeGaussiansCountCUDA);
  m.def("rasterize_gaussians_indexed", &RasterizeGaussiansIndexedCUDA);
  m.def("rasterize_gaussians_depth", &RasterizeGaussiansDepthCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("rasterize_gaussians_backward_indexed", &RasterizeGaussiansBackwardIndexedCUDA);
  m.def("mark_visible", &markVisible);
}