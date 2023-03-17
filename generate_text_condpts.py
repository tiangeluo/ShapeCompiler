import argparse
from pathlib import Path
from tqdm import tqdm

import torch

from einops import repeat


from PIL import Image
from torchvision.utils import make_grid, save_image


from core_codes.shapecompiler import PointVQVAE, ShapeCompiler
from core_codes.tokenizer import tokenizer
from IPython import embed
import os
from pytorch3d.io import save_ply, load_ply


parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type = str, required = True,
                    help='path to your trained ShapeCompiler')

parser.add_argument('--pts_path', type = str, required=True,
                    help='path to your input point cloud')

parser.add_argument('--num_txts', type = int, default = 128, required = False,
                    help='number of texts you want to generate')

parser.add_argument('--batch_size', type = int, default = 16, required = False,
                    help='generation batch size')

parser.add_argument('--top_k', type = float, default = 0.9, required = False,
                    help='top k filter threshold')

parser.add_argument('--outputs_dir', type = str, default = './outputs/shapecompiler_outputs', required = False,
                    help='output directory')

parser.add_argument('--save_name', type = str, default = 'test1', help = 'save name')

parser.add_argument('--inverse', action= 'store_true',
                    help='inverse the point clouds orientation')

args = parser.parse_args()

def normalize_points_torch(points):
    """Normalize point cloud

    Args:
        points (torch.Tensor): (batch_size, num_points, 3)

    Returns:
        torch.Tensor: normalized points

    """
    assert points.dim() == 3 and points.size(2) == 3
    centroid = points.mean(dim=1, keepdim=True)
    points = points - centroid
    norm, _ = points.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)
    new_points = points / norm
    return new_points

def exists(val):
    return val is not None

model_path = Path(os.path.join('./outputs/shapecompiler_models',args.model_path))

assert model_path.exists(), 'trained shape compiler must exist'

# load pre-trained model
load_obj = torch.load(str(model_path))
shapecompiler_params, pointvqvae_params, weights, vae_class_name, version = load_obj.pop('hparams'), load_obj.pop('vae_params'), load_obj.pop('weights'), load_obj.pop('vae_class_name', None), load_obj.pop('version', None)
pointvqvae = PointVQVAE(**pointvqvae_params)
shapecompiler = ShapeCompiler(vae = pointvqvae, **shapecompiler_params).cuda()
shapecompiler.load_state_dict(weights)


save_dir = os.path.join(args.outputs_dir,'pts2text_'+args.save_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

pc_path = os.path.join(args.pts_path)
pc = load_ply(pc_path)
if args.inverse:
    pc[0][:,2] *= -1
save_ply(os.path.join(save_dir, 'input_pc.ply'), pc[0])
points = normalize_points_torch(pc[0].unsqueeze(0)).cuda()
points = repeat(points, '() n c -> b n c', b = args.num_txts)

text_token_list = []
text_list = []
for pts_chunk in tqdm(points.split(args.batch_size), desc = f'generating text for - {pc_path}'):
    text_tokens, gen_texts = shapecompiler.generate_text_condpts(tokenizer, pts_chunk, filter_thres = args.top_k)
    text_token_list.append(text_tokens)
    text_list.append(gen_texts)

with open(os.path.join(save_dir, 'gen_text.txt'), 'w') as f:
    for i in text_list:
        for j in i:
            f.writelines(j + '\n')