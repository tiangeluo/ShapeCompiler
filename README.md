# Neural Shape Compiler: A Unified Framework for Transforming between Text, Point Cloud, and Program

In this repo, we release our data, codes, and pre-trained models for this Neural Shape Compiler [[project page](https://tiangeluo.github.io/projectpages/shapecompiler.html)]. 



We released our data, model file, pre-trained model, and some of the inference codes. More inference codes, test codes and training codes are on the way. 


## Installation

Please proceed 1~4 steps below:

1. ```sh
   git clone --recurse-submodules https://github.com/tiangeluo/ShapeCompiler.git
   
   conda create --name shapecompiler python=3.8
   conda activate shapecompiler
   ```

2. install [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorch3D](https://pytorch3d.org/). 

   ```sh
   # my install commands
   pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
   
   conda install -c fvcore -c iopath -c conda-forge fvcore iopath
   conda install -c bottler nvidiacub
   pip install "git+https://github.com/facebookresearch/pytorch3d.git"
   ```

   

3. ```sh
   cd ShapeCompiler
   python setup.py install
   # my gcc version is 8.2.0 for compiling cuda operators
   bash compile.sh
   ```

4. download pre-trained model `shapecompiler.pt` from [GoogleDrive](https://drive.google.com/file/d/1Y__4AIMmrM9ECasWw5w0qJiE_DjxjmwW/view?usp=sharing) , and move it to `ShapeCompiler/outputs/shapecompiler_models`

## Inference

```bash
# to generate point clouds conditional on text
# results will be saved in ./outputs/shapecompiler_outputs/text2pts_test1
python generate_pts_condtext.py --model_path ./shapecompiler.pt --text 'a chair has armrests, with slats between legs' --save_name 'test1' 

# to generate text conditional on point clouds
# results will be saved in ./outputs/shapecompiler_outputs/pts2text_test1
python generate_text_condpts.py --model_path ./shapecompiler.pt --pts_path './assets/example_chair.ply' --save_name 'test1' 

# note that point clouds extract from ShapeNet has different orientation as we train ShapeCompiler
# assume point clouds has: pc.shape = [2048, 3]. you need to turn pc[:,2] = -1*pc[:,2]
# you can add flag --inverse in your command line to conduct pc[:,2] = -1*pc[:,2] 
# if you are not confident if the shape orientation is correct, please visualize ./assets/example_chair.ply

# to generate programs conditional on point clouds
# generated program parameters, program text, voxels, and extracted point clouds will be saved in ./outputs/shapecompiler_outputs/pts2pgm_test1
python generate_pgm_condpts.py --model_path ./shapecompiler.pt --pts_path './assets/example_chair.ply' --save_name 'test1' 

```



## Pre-trained checkpoints

| Description                                                  | Link                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Shape Compiler, training with all the data mentioned in paper | [Download (1.49GB)](https://drive.google.com/file/d/1Y__4AIMmrM9ECasWw5w0qJiE_DjxjmwW/view?usp=sharing) |
| PointVQVAE, training with ABO, ShapeNet, Program objects     | [Download (107.3 MB)](https://drive.google.com/file/d/1Y1PSnSukRwub1tJRix4NEO3_aKl-Xx65/view?usp=share_link) |
| PointVQVAE, training with ShapeNet objects                   | Stay Tuned                                                   |
| PointVQVAE, training with ABO, ShapeNet, Program, Objaverse objects | Stay Tuned                                                   |

## Data

Our [shape, structural description] paired data is stored under `/data` as pickle files and be loaded via `data= pickle.load(open('abo_text_train.pkl','rb'))`. Each pickle file contains `num of indices` and `pcs_name`. You can accesee the text annotation by index (e.g., `data[10]`) and its correspondence point cloud file name (e.g., `data['pcs_name'][10]`). 

Please also cite [ABO](https://arxiv.org/abs/2110.06199), [ShapeNet](https://arxiv.org/abs/1512.03012), [Text2Shape](http://text2shape.stanford.edu/), and [ShapeGlot](https://arxiv.org/abs/1905.02925), if you use our caption data along with the objects provided in their datasets.

## Other Interesting Ideas

- Text->3D: [Text2Shape](http://text2shape.stanford.edu/), [DreamField](https://ajayj.com/dreamfields), [DreamFusion](https://dreamfusion3d.github.io/), [Shape IMLE](https://github.com/liuzhengzhe/Towards-Implicit-Text-Guided-Shape-Generation), [CLIPForge](https://github.com/AutodeskAILab/Clip-Forge), [Magic3D](https://deepimagination.cc/Magic3D/), [ShapeCrafter](https://arxiv.org/abs/2207.09446), [Shape2VecSet](https://arxiv.org/abs/2301.11445), [MeshDiffusion](https://openreview.net/pdf?id=0cpM2ApF9p6)

- 3D->Program: [ShapeProgram](http://shape2prog.csail.mit.edu/), [ShapeAssembly](https://github.com/rkjones4/ShapeAssembly), [LegoAssembly](https://cs.stanford.edu/~rcwang/projects/lego_manual/)

- 3D->Text: [Scan2cap](https://arxiv.org/abs/2012.02206)

## Acknowledgement
We thank the below open-resource projects and codes.

- [PyTorch](https://www.github.com/pytorch/pytorch) and [PyTorch3D](https://github.com/facebookresearch/pytorch3d).
- Our codebase builds heavily on https://github.com/lucidrains/DALLE-pytorch. 
- PointVQVAE implementation is built based on Shaper developed by [Jiayuan](https://github.com/Jiayuan-Gu).
- We follow this [script](https://github.com/zekunhao1995/PointFlowRenderer) to render our point clouds with [Misuba](http://www.mitsuba-renderer.org/).

## BibTex
If you find our work or repo helpful, we are happy to receive a citation.

```
@article{luo2022neural,
      title={Neural Shape Compiler: A Unified Framework for Transforming between Text, Point Cloud, and Program},
      author={Luo, Tiange and Lee, Honglak and Johnson, Justin},
      journal={arXiv preprint arXiv:2212.12952},
      year={2022}
}
```
