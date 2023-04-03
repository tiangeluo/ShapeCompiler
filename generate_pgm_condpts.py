import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from einops import repeat

from core_codes.shapecompiler import PointVQVAE, ShapeCompiler
from IPython import embed
import os
from pytorch3d.io import save_ply, load_ply
from pytorch3d.ops import cubify, sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

import h5py
import numpy as np

import sys
sys.path.append('./shape2prog')
from misc import execute_shape_program
from interpreter import Interpreter
from programs.loop_gen import translate, rotate, end

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type = str, required = True,
                    help='path to your trained ShapeCompiler')

parser.add_argument('--pts_path', type = str, required=True,
                    help='path to your input point cloud')

parser.add_argument('--num_programs', type = int, default = 128, required = False,
                    help='number of programs you want to generate')

parser.add_argument('--batch_size', type = int, default = 16, required = False,
                    help='generation batch size')

parser.add_argument('--top_k', type = float, default = 0.9, required = False,
                    help='top k filter threshold')

parser.add_argument('--outputs_dir', type = str, default = './outputs/shapecompiler_outputs', required = False,
                    help='output directory')

parser.add_argument('--save_name', type = str, default = 'test1', help = 'save name')


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


save_dir = os.path.join(args.outputs_dir, 'pts2pgm_'+args.save_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

pc_path = os.path.join(args.pts_path)
pc = load_ply(pc_path)
points = normalize_points_torch(pc[0].unsqueeze(0)).cuda()
points = repeat(points, '() n c -> b n c', b = args.num_programs)

out_pgms = []
out_params = []
for pts_chunk in tqdm(points.split(args.batch_size), desc = f'generating programs for - {pc_path}'):
    output = shapecompiler.generate_pgm_condpts(pts_chunk, filter_thres = args.top_k)
    out_pgms.append(output[0])
    out_params.append(output[1])

out_pgms = torch.cat(out_pgms)
out_params = torch.cat(out_params)
## the below two for loops remove outliers; however, the generated output should generally have no outliers.
for i in range(out_pgms.shape[0]):
    out_pgm_cur = out_pgms[i]
    less_zero = torch.sum(out_pgm_cur < 0)
    be_up = torch.sum(out_pgm_cur > 20)
    #print('output:%d, less_zero:%d, bigger_up:%d'%(i, less_zero, be_up))
    out_pgms[i][out_pgm_cur < 0] = 0
    out_pgms[i][out_pgm_cur >20] = 20

for i in range(out_params.shape[0]):
    out_param_cur = out_params[i]
    less_zero = torch.sum(out_param_cur < -27)
    be_up = torch.sum(out_param_cur > 29)
    #print('output:%d, less_zero:%d, bigger_up:%d'%(i, less_zero, be_up))
    out_params[i][out_param_cur < -27] = -27
    out_params[i][out_param_cur > 29] = 29

out_pgm=out_pgms.reshape(-1,10,3)
out_param = out_params.reshape(-1, 10, 3 ,7)

save_obj = {
    'pgm': out_pgm,
    'param': out_param,
}
torch.save(save_obj, os.path.join(save_dir,'generated_program_parameters.pt'))
save_ply(os.path.join(save_dir,'input_pc.ply'), points[0])

# reconstruct voxels from programs, and save them to h5 file
out_pgms = out_pgm.cpu().numpy()
out_params = out_param.cpu().numpy()
num_shapes = out_pgms.shape[0]
voxels=[]
for i in range(num_shapes):
    try:
        data = execute_shape_program(out_pgms[i], out_params[i])
    except:
        print('render program wrong', i)
        continue
    voxels.append(data.reshape((1, 32, 32, 32)))
if len(voxels) == 0:
    print('All generated programs failed.')
else:
    # [batch_size, 32, 32, 32]
    h5_file = h5py.File(os.path.join(save_dir,'generated_voxel.h5'), 'w')
    h5_file['voxel'] = np.concatenate(voxels, 0)
    h5_file.close()

# extract point clouds from voxels, and save
ours_pc_list = []
for voxel in voxels:
    tmp_mesh = cubify(torch.Tensor(voxel).cuda(),0.5)
    # pts.size == [1, 10000, 3]
    pts = sample_points_from_meshes(tmp_mesh)
    perm_axis_pts = torch.zeros(pts.shape).cuda()
    perm_axis_pts[:, :, 0] = pts[:, :, 0]
    perm_axis_pts[:, :, 1] = pts[:, :, 2]
    perm_axis_pts[:, :, 2] = -1*pts[:, :, 1]
    ours_pc_list.append(normalize_points_torch(perm_axis_pts))
gen_pts = torch.cat(ours_pc_list, dim = 0)
ori_pts = points[:gen_pts.shape[0]]
cd_dis = chamfer_distance(gen_pts, ori_pts, batch_reduction = None)[0]
sort_idx = np.argsort(cd_dis.cpu().numpy())
save_dir_ply = os.path.join(save_dir,'generated_pc')
if not os.path.exists(save_dir_ply):
    os.mkdir(save_dir_ply)
for i in range(sort_idx.shape[0]):
    save_ply(os.path.join(save_dir_ply,'%03d.ply'%i), gen_pts[sort_idx[i]])

# save programs
save_dir_pgm = os.path.join(save_dir,'generated_program')
if not os.path.exists(save_dir_pgm):
    os.mkdir(save_dir_pgm)
interpreter = Interpreter(translate, rotate, end)
for i in range(sort_idx.shape[0]):
    idx = sort_idx[i]
    program = interpreter.interpret(out_pgms[idx], out_params[idx])
    save_file = os.path.join(save_dir_pgm, '%03d.txt'%i)
    with open(save_file, 'w') as out:
        out.write(program)

print('save_dir:%s'%save_dir)
