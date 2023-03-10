# Neural Shape Compiler: A Unified Framework for Transforming between Text, Point Cloud, and Program

In this repo, we release our data, codes, and pre-trained models for [paper](https://arxiv.org/abs/2212.12952). Codes and pre-trained models are coming soon.

## Data

Our [shape, structural description] paired data is stored under `/data` as pickle files and be loaded via `data= pickle.load(open('abo_text_train.pkl','rb'))`. Each pickle file contains `num of indices` and `pcs_name`. You can accesee the text annotation by index (e.g., `data[10]`) and its correspondence point cloud file name (e.g., `data['pcs_name'][10]`). 

If you use our data, please also cite [ABO](https://arxiv.org/abs/2110.06199), [ShapeNet](https://arxiv.org/abs/1512.03012), [Text2Shape](http://text2shape.stanford.edu/), and [ShapeGlot](https://arxiv.org/abs/1905.02925).



## Other Interesting Ideas

- Text->3D: [Text2Shape](http://text2shape.stanford.edu/), [DreamFusion](https://dreamfusion3d.github.io/), [DreamField](https://ajayj.com/dreamfields), [Shape IMLE](https://github.com/liuzhengzhe/Towards-Implicit-Text-Guided-Shape-Generation), [CLIPForge](https://github.com/AutodeskAILab/Clip-Forge), [Magic3D](https://deepimagination.cc/Magic3D/), [ShapeCrafter](https://arxiv.org/abs/2207.09446), [Shape2VecSet](https://arxiv.org/abs/2301.11445)

- 3D->Program: [ShapeProgram](http://shape2prog.csail.mit.edu/), [ShapeAssembly](https://github.com/rkjones4/ShapeAssembly)

- 3D->Text: [Scan2cap](https://arxiv.org/abs/2012.02206)

## Acknowledgement
We thank the below open-resource projects and codes.

- Our codebase builds heavily on https://github.com/lucidrains/DALLE-pytorch. 
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
