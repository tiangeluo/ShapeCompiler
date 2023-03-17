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
from pytorch3d.io import save_ply
import time
import numpy as np

# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type = str, required = True,
                    help='path to your trained ShapeCompiler')

parser.add_argument('--text', type = str, required = True,
                    help='your text prompt')

parser.add_argument('--num_pointclouds', type = int, default = 128, required = False,
                    help='number of point clouds you want to generate')

parser.add_argument('--batch_size', type = int, default = 16, required = False,
                    help='generation batch size')

parser.add_argument('--top_k', type = float, default = 0.9, required = False,
                    help='top k filter threshold')

parser.add_argument('--outputs_dir', type = str, default = './outputs/shapecompiler_outputs', required = False,
                    help='output directory')

parser.add_argument('--save_name', type = str, default = 'test1', help = 'save name')

parser.add_argument('--gentxt', dest='gentxt', action='store_true')

args = parser.parse_args()

# helper fns

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

texts = args.text.split('|')

for j, text in tqdm(enumerate(texts)):
    # this function performs unconditional text generation, 
    # and then do text-conditional pointcloud generation;
    # achieve unconditional pointcloud generation.
    if args.gentxt:
        text_tokens, gen_texts = shapecompiler.generate_texts(tokenizer, text=text, filter_thres = args.top_k)
        text = gen_texts[0]
    else:
        text_tokens = tokenizer.tokenize([text], shapecompiler.text_seq_len).cuda()

    text_tokens = repeat(text_tokens, '() n -> b n', b = args.num_pointclouds)

    outputs = []

    start_time = time.time()
    for text_chunk in tqdm(text_tokens.split(args.batch_size), desc = f'generating point clouds for - {text}'):
        output = shapecompiler.generate_pts_condtext(text_chunk, filter_thres = args.top_k)
        outputs.append(output)
    print('total_time:', time.time() - start_time)

    outputs = torch.cat(outputs)

    # to avoid input text too long to create a valid file in Linux
    file_name = text.replace(' ','_')[:100]
    outputs_dir = os.path.join(args.outputs_dir, 'text2pts_'+args.save_name)
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)

    for i, image in tqdm(enumerate(outputs), desc = 'saving point clouds'):
        save_ply(os.path.join(outputs_dir,'%04d.ply'%i),image)
    with open(os.path.join(outputs_dir,'text_prompt.txt'), 'w') as f:
        f.write(text)

    print(f'created {args.num_pointclouds} point clouds at "{str(outputs_dir)}"')
