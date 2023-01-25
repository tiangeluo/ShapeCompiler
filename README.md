# Neural Shape Compiler: A Unified Framework for Transforming between Text, Point Cloud, and Program

In this repo, we release our data, codes, and pre-trained models for [paper](https://arxiv.org/abs/2212.12952). Codes and pre-trained models are coming soon.

## Data

Our [shape, structural description] paired data is stored under `/data` as pickle files and be loaded via `data= pickle.load(open('abo_text_train.pkl','rb'))`. Each pickle file contains `num of indices` and `pcs_name`. You can accesee the text annotation by index (e.g., `data[10]`) and its correspondence point cloud file name (e.g., `data['pcs_name'][10]`). 

If you use our data, please also cite (ABO)[https://arxiv.org/abs/2110.06199], (ShapeNet)[https://arxiv.org/abs/1512.03012], (Text2Shape)[Text2Shape: Generating Shapes from Natural Language by Learning Joint Embeddings], and (ShapeGlot)[https://arxiv.org/abs/1905.02925].


## Acknowledgement
- Our codebase builds heavily on https://github.com/lucidrains/DALLE-pytorch. We appreciate [lucidrains](https://github.com/lucidrains) for open-sourcing.
- We follow this [script](https://github.com/zekunhao1995/PointFlowRenderer) to render our point clouds with [Misuba](http://www.mitsuba-renderer.org/).

## BibTex

```
@article{luo2022neural,
      author = {Luo, Tiange and Lee, Honglak and Johnson, Justin},
      title = {Neural Shape Compiler: A Unified Framework for Transforming between Text, Point Cloud, and Program},
      journal = {Workshop on Learning to Generate 3D Shapes and Scenes at ECCV},
      year = {2022},
}
```

