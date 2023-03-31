# Neural Shape Compiler: A Unified Framework for Transforming between Text, Point Cloud, and Program

In this repo, we release our data, codes, and pre-trained models for this Neural Shape Compiler [project](https://tiangeluo.github.io/projectpages/shapecompiler.html). 



We released our data, model file, pre-trained model, and some of the inference codes. More inference codes, test codes and training codes are on the way. 

## Data

Our [shape, structural description] paired data is stored under `/data` as pickle files and be loaded via `data= pickle.load(open('abo_text_train.pkl','rb'))`. Each pickle file contains `num of indices` and `pcs_name`. You can accesee the text annotation by index (e.g., `data[10]`) and its correspondence point cloud file name (e.g., `data['pcs_name'][10]`). 

If you use our data, please also cite [ABO](https://arxiv.org/abs/2110.06199), [ShapeNet](https://arxiv.org/abs/1512.03012), [Text2Shape](http://text2shape.stanford.edu/), and [ShapeGlot](https://arxiv.org/abs/1905.02925).



## Installation

There are several steps to do before running:

1. running `bash compiler.sh` to install some CUDA extensions for PointVQVAE
2. install `Pytorch3D` by following steps provided in https://pytorch3d.org/
3. `python setup.py install`
4. download pre-trained model `shapecompiler.pt` from [GoogleDrive](https://drive.google.com/file/d/1Y__4AIMmrM9ECasWw5w0qJiE_DjxjmwW/view?usp=sharing) , and move it into `ShapeCompiler/outputs/shapecompiler_models`

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

# to generate programs confitional on point clouds
# results will be saved in ./outputs/shapecompiler_outputs/pts2pgm_test1
# the code to print out programs and visualize the red voxels will be released later
python generate_pgm_condpts.py --model_path ./shapecompiler.pt --pts_path './assets/example_chair.ply' --save_name 'test1' 

```



## Other Interesting Ideas

- Text->3D: [Text2Shape](http://text2shape.stanford.edu/), [DreamFusion](https://dreamfusion3d.github.io/), [DreamField](https://ajayj.com/dreamfields), [Shape IMLE](https://github.com/liuzhengzhe/Towards-Implicit-Text-Guided-Shape-Generation), [CLIPForge](https://github.com/AutodeskAILab/Clip-Forge), [Magic3D](https://deepimagination.cc/Magic3D/), [ShapeCrafter](https://arxiv.org/abs/2207.09446), [Shape2VecSet](https://arxiv.org/abs/2301.11445), [MeshDiffusion](https://openreview.net/pdf?id=0cpM2ApF9p6)

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
