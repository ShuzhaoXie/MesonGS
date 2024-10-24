# MesonGS: Post-training 3D Gaussian Compression [ECCV 2024]
Shuzhao Xie,
Weixiang Zhang, 
<a href="https://www.chentang.cc">Chen Tang</a>,
<a href="https://bbaaii.github.io/">Yunpeng Bai</a>,
<a href="">Rongwei Lu</a>,
Shijia Ge,
<a href="http://zwang.inflexionlab.org">Zhi Wang</a> 
<br><br>
[Webpage](https://shuzhaoxie.github.io/mesongs/) | [Paper](https://shuzhaoxie.github.io/data/24-eccv-mesongs.pdf) <br>

<!-- ![Teaser image](assets/teaser.png) -->

This repository contains the official authors implementation associated with the paper "MesonGS: Post-training Compression of 3D Gaussians via Efficient Attribute Transformation".


<a href="https://www.tsinghua.edu.cn/"><img height="100" src="assets/thu_logo.png"></a>
<a href="https://www.pcl.ac.cn/"><img height="100" src="assets/pcl_logo.png"> </a>
<a href="https://www.cuhk.edu.hk/"><img height="100" src="assets/cuhk_logo.png"> </a>
<a href="https://www.utexas.edu/"><img height="100" src="assets/ut_logo.png"> </a> 



Abstract: *3D Gaussian Splatting demonstrates excellent quality and speed in novel view synthesis. Nevertheless, the huge file size of the 3D Gaussians presents challenges for transmission and storage. Current works design compact models to replace the substantial volume and attributes of 3D Gaussians, along with intensive training to distill information. These endeavors demand considerable training time, presenting formidable hurdles for practical deployment. To this end, we propose MesonGS, a codec for post-training compression of 3D Gaussians. Initially, we introduce a measurement criterion that considers both view-dependent and view-independent factors to assess the impact of each Gaussian point on the rendering output, enabling the removal of insignificant points. Subsequently, we decrease the entropy of attributes through two transformations that complement subsequent entropy coding techniques to enhance the file compression rate. More specifically, we first replace rotation quaternions with Euler angles; then, we apply region adaptive hierarchical transform to key attributes to reduce entropy. Lastly, we adopt finer-grained quantization to avoid excessive information loss. Moreover, a well-crafted finetune scheme is devised to restore quality. Extensive experiments demonstrate that MesonGS significantly reduces the size of 3D Gaussians while preserving competitive quality.*



## 1. Cloning the Repository

```shell
git clone https://github.com/ShuzhaoXie/MesonGS.git
```

## 2. Install

### 2.1 Hardware and Software Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- Ubuntu >= 18.04
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we *recommend* Visual Studio 2019 for Windows)
- CUDA >= 11.6
- C++ Compiler and CUDA SDK must be compatible

### 2.2 Setup

Our provided install method is based on Conda package and environment management:

CUDA 11.6/11.8, GPU 3090/4090/V100: 

```shell
sudo apt install zip unzip
conda env create --file environment.yml
conda activate mesongs
pip install plyfile tqdm einops scipy open3d trimesh Ninja seaborn loguru pandas torch_scatter mediapy
```

CUDA 12.1/12.4, GPU 3090/4090:

```shell
sudo apt install zip unzip
conda create -n mesongs python=3.10
conda activate mesongs
pip install torchaudio==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install plyfile tqdm einops scipy open3d trimesh Ninja seaborn loguru pandas torch_scatter mediapy
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/weighted_distance
```

### 2.3 Enviroment Variables
* Replace the `MAIN_DIR` in [utils/system_utils.py](utils/system_utils.py) with your dir path.
* prepare directories: 
  ```
  ## cd to your path of mesongs
  mkdir output
  mkdir data
  mkdir exp_data
  mkdir exp_data/csv
  ```

### 2.4 Preparing dataset and pre-trained 3D Gaussians
You can download a sample checkpoint of `mic` scene from [here [68 MB]](https://drive.google.com/file/d/1VqDNh7lHraWrA7uj8Dhw62pyZgr_kzLy/view?usp=drive_link). Then:
1. Unzip and put the checkpoint directory into the `output` directory. 

    ```bash
    .
    ├── cameras.json
    ├── cfg_args
    ├── input.ply
    └── point_cloud
        └── iteration_30000
            └── point_cloud.ply
    ```
2. Put the datasets into the `data` directory. Our implementation only supports the datasets listed below.
    ```bash
    .
    ├── 360_v2
    │   ├── bicycle
    │   ├── bonsai
    │   ├── counter
    │   ├── flowers
    │   ├── garden
    │   ├── kitchen
    │   ├── room
    │   ├── stump
    │   └── treehill
    ├── db
    │   ├── drjohnson
    │   └── playroom
    ├── nerf_synthetic
    │   ├── chair
    │   ├── drums
    │   ├── ficus
    │   ├── hotdog
    │   ├── lego
    │   ├── materials
    │   ├── mic
    │   ├── README.txt
    │   └── ship
    └── tandt
        ├── train
        └── truck
    ```

## 3. Running

To run the MesonGS, using:

```bash
bash scripts/mesongs_block.sh
```

* Set `--iteration` to `0` for compression without finetuning. 
* Add `--skip_post_eval` to skip tedious testing process.
* Check `scripts` dir for more examples.


### 3.1 Rendering Compressed File
To render compressed file, take `mic` as an example, run:
```bash
MAINDIR=/your/path/to/mesongs
DATADIR=/your/path/to/data
CKPT=meson_mic
SCENENAME=mic

python render.py -s $DATADIR/nerf_synthetic/$SCENENAME \
    --given_ply_path $MAINDIR/output/$CKPT/point_cloud/iteration_0/pc_npz/bins.zip \
    --eval --skip_train -w \
    --dec_npz \
    --scene_name $SCENENAME \
    --csv_path $MAINDIR/exp_data/csv/test_$CKPT.csv \
    --model_path $MAINDIR/output/$CKPT
```

### 3.2 Evaluation and Comparison
Please consider using the [`c1`](configs/c1.json) and [`c3`](configs/c3.json) configs. You can find the corresponding evaluation results at the [`results`](results/) directory. 
We also share these results with the [3DGS compression survey](https://w-m.github.io/3dgs-compression-survey/). This survey is wonderful and presents lots of baselines.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">4. BibTeX</h2>
    <pre>
    <code>@inproceedings{xie2024mesongs,
        title={MesonGS: Post-training Compression of 3D Gaussians via Efficient Attribute Transformation},
        author={Xie, Shuzhao and Zhang, Weixiang and Tang, Chen and Bai, Yunpeng and Lu, Rongwei and Ge, Shijia and Wang, Zhi},
        booktitle={European Conference on Computer Vision},
        year={2024},
        organization={Springer}
    }
    </code>
    </pre>   
  </div>
</section>



## 5. TODO
- [x] Upload the version that configed by number of blocks instead of the length.

## 6. Contributions
Some source code of ours is borrowed from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [3DAC](https://github.com/fatPeter/ThreeDAC), [c3dgs](https://github.com/KeKsBoTer/c3dgs), [LightGuassian](https://github.com/VITA-Group/LightGaussian), and [ACRF](https://github.com/fatPeter/ACRF). We sincerely appreciate the excellent works of these authors.

## 7. Funding and Acknowledgments

This work is supported in part by National Key Research and Development Project of China (Grant No. 2023YFF0905502) and Shenzhen Science and Technology Program (Grant No. JCYJ20220818101014030). We thank anonymous reviewers for their valuable advice and [JiangXingAI](https://www.jiangxingai.com/) for sponsoring the research.