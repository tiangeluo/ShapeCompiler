# Neural Shape Compiler

In this repo, we release our data, codes, and pre-trained models for [paper](https://arxiv.org/abs/2212.12952). Codes and pre-trained models are coming soon.



Our shape-text paired data is stored under `/data` as pickle files and be loaded via `data= pickle.load(open('abo_text_train.pkl','rb'))`. Each pickle file contains `num of indices` and `pcs_name`. You can accesee the text annotation by index (e.g., `data[10]`) and its correspondence point cloud file name (e.g., `data['pcs_name'][10]`). 
