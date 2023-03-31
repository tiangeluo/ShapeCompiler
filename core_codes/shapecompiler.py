from math import log2, sqrt
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np

from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

from dalle_pytorch import distributed_utils
from dalle_pytorch.vae import OpenAIDiscreteVAE, VQGanVAE
from dalle_pytorch.transformer import Transformer, DivideMax

# helpers
import torch
import torch.nn as nn

import sys
from IPython import embed

sys.path.insert(0, '..')
from shaper.models.pointnet2.modules import PointNetSAModule, PointnetFPModule

def set_bn(module, momentum):
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = momentum


def xavier_uniform(module):
    if module.weight is not None:
        nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class always():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return self.val

def is_empty(t):
    return t.nelement() == 0

def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]

def prob_mask_like(shape, prob, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class SharedEmbedding(nn.Embedding):
    def __init__(self, linear, start_index, end_index, **kwargs):
        super().__init__(end_index - start_index, linear.weight.shape[1], **kwargs)
        del self.weight

        self.linear = linear
        self.start_index = start_index
        self.end_index = end_index

    def forward(self, input):
        return F.embedding(
            input, self.linear.weight[self.start_index:self.end_index], self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

class VQVAE_Decoder_depth3(nn.Module):
    def __init__(self, feat_dims, codebook_dim=512, final_dim=2048):
        super(VQVAE_Decoder_depth3, self).__init__()
        self.dim = codebook_dim
        self.folding1 = nn.Sequential(
            nn.Conv1d(self.dim, 4*self.dim, 1),
            nn.BatchNorm1d(4*self.dim),
            nn.ReLU(),
            nn.Conv1d(4*self.dim, 4*self.dim, 1),
            nn.BatchNorm1d(4*self.dim),
            nn.ReLU(),
            nn.Conv1d(4*self.dim, self.dim, 1),
        )
         
        self.folding2 = nn.Sequential(
            nn.Conv1d(self.dim, 4*self.dim, 1),
            nn.BatchNorm1d(4*self.dim),
            nn.ReLU(),
            nn.Conv1d(4*self.dim, 4*self.dim, 1),
            nn.BatchNorm1d(4*self.dim),
            nn.ReLU(),
            nn.Conv1d(4*self.dim, self.dim, 1),
        )
        self.folding3 = nn.Sequential(
            nn.Conv1d(self.dim, 4*self.dim, 1),
            nn.BatchNorm1d(4*self.dim),
            nn.ReLU(),
            nn.Conv1d(4*self.dim, 4*self.dim, 1),
            nn.BatchNorm1d(4*self.dim),
            nn.ReLU(),
            nn.Conv1d(4*self.dim, self.dim, 1),
        )
        self.our_end = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(self.dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, final_dim*3, 1)
        )

    def forward(self, x):
        folding_result1 = self.folding1(x)              # (#batch_size, #codebook_dim: 512, #part_embeding_num 128)
        x = x+folding_result1
        folding_result2 = self.folding2(x)              # (#batch_size, #codebook_dim: 512, #part_embeding_num 128)
        x = x+folding_result2
        folding_result3 = self.folding3(x)              # (#batch_size, #codebook_dim: 512, #part_embeding_num 128)
        x = x+folding_result3
        max_feature = torch.max(x, -1, keepdim=True)[0] # (#batch_size, #codebook_dim: 512, 1)
        output = self.our_end(max_feature)              # (#batch_size, 3*2048, 1)
        return output          


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        # [8, 512, 128, 1]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        # [8, 128, 1, 512]
        input_shape = inputs.shape
        
        # Flatten input
        # flat_input = inputs.view(-1, self._embedding_dim)
        # [8*128, 512]
        flat_input = inputs.view(-1, self._num_embeddings)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        # [4, 128, 1, 512]
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        unique_arr = []
        for j in range(input_shape[0]):
            unique_arr.append(len(torch.unique(torch.argmax(encodings.view(input_shape), -1).squeeze()[j])))
        perplexity = torch.mean(torch.Tensor(unique_arr))
        
        # convert quantized from BHWC -> BCHW
        #quantized.permute: [4, 512, 128, 1]
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices.reshape(inputs.shape[0],-1)

class PointVQVAE(nn.Module):
    def __init__(
        self,
        num_tokens = 512,
        codebook_dim = 512,
        num_layers = 3,
        num_resnet_blocks = 0,
        hidden_dim = 64,
        channels = 3,
        smooth_l1_loss = False,
        temperature = 0.9,
        straight_through = False,
        kl_div_loss_weight = 0.,
        normalization = ((0.5,) * 3, (0.5,) * 3),
        dim1 = 16,
        dim2 = 32,
        radius= 0.4,
        final_points = 16,
        final_dim = 2048,
        vae_type = 1,
        vae_encode_type = 1,
        image_size = 256,
    ):
        super().__init__()
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.pts_size = image_size
        # self.codebook = nn.Embedding(num_tokens, codebook_dim)
        self.codebook_dim = codebook_dim
        self.final_dim = final_dim

        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        self.final_points = final_points
        self.radius = radius
        self.quantize_layer = VectorQuantizer(num_tokens, codebook_dim)
        self.decoder = VQVAE_Decoder_depth3(self.final_points, codebook_dim, final_dim)

        in_channels = 6

        num_centroids=(512, self.final_points)
        radius=(0.1, 0.4)
        num_neighbours=(64, 512)
        sa_channels=((512, 512), (512, num_tokens))
        use_xyz=True
        num_sa_layers = len(num_centroids)

        feature_channels = in_channels - 3
        self.sa_modules = nn.ModuleList()
        for ind in range(num_sa_layers):
            sa_module = PointNetSAModule(in_channels=feature_channels,
                                         mlp_channels=sa_channels[ind],
                                         num_centroids=num_centroids[ind],
                                         radius=radius[ind],
                                         num_neighbours=num_neighbours[ind],
                                         use_xyz=use_xyz)
            self.sa_modules.append(sa_module)
            feature_channels = sa_channels[ind][-1]
        self.reset_parameters()


        self.normalization = normalization

        self._register_external_parameters()

    def reset_parameters(self):
        for sa_module in self.sa_modules:
            sa_module.reset_parameters(xavier_uniform)
        # self.mlp_seg.reset_parameters(xavier_uniform)
        set_bn(self, momentum=0.01)

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if (
                not distributed_utils.is_distributed
                or not distributed_utils.using_backend(
                    distributed_utils.DeepSpeedBackend)
        ):
            return

        deepspeed = distributed_utils.backend.backend_module
        # deepspeed.zero.register_external_parameter(self, self.codebook.weight)

    def norm(self, pts):
        if not exists(self.normalization):
            return pts

        means, stds = map(lambda t: torch.as_tensor(t).to(pts), self.normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c () ()'), (means, stds))
        pts = pts.clone()
        pts.sub_(means).div_(stds)
        return pts

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, pts):

        logits = self(pts, return_logits = True)
        _, _, _, indices = self.quantize_layer(logits.unsqueeze(-1))
        return indices


    def decode(
        self,
        indices
    ):

        flat_indices = indices.reshape(-1,1)
        encodings = torch.zeros(flat_indices.shape[0], self.quantize_layer._num_embeddings, device=indices.device)
        encodings.scatter_(1, flat_indices, 1)
        quantized = torch.matmul(encodings, self.quantize_layer._embedding.weight).view(indices.shape[0],indices.shape[1],1,-1).permute(0,3,1,2)
        pcs = self.decoder(quantized.squeeze(-1)).reshape(quantized.shape[0],-1,3)
        return pcs

    def get_encoding(self, pts):
        logits = self(pts, return_logits = True)
        _, quantized, _, _ = self.quantize_layer(logits.unsqueeze(-1))
        return quantized

    def forward(
        self,
        pts,
        return_loss = False,
        return_recons = False,
        return_logits = False,
        return_detailed_loss = False,
        temp = None,
        epoch = 0,
    ):

        points = pts.transpose(1,2)
        xyz = points
        feature = points
        for sa_module in self.sa_modules:
            xyz, feature = sa_module(xyz, feature)

        if return_logits:
            return feature # return logits for getting hard pts indices for DALL-E training

        vq_loss, sampled, perplexity, _ = self.quantize_layer(feature.unsqueeze(-1))
        
        out = self.decoder(sampled.squeeze(-1))

        if not return_loss:
            return out

        # if not return_recons:
        #     return loss

        # if not return_detailed_loss:
        #     return loss, out, perplexity
        # else:
        #     return loss, cd_loss, emd_loss, vq_loss, out, perplexity
def decode_pgcode2(pg_codes):
    # params [-27, 29] to --> [0, 56]
    # pgms [0, 20] to --> [57, 77]
    # bs = pg_codes.shape[0]
    pgm_idx = (torch.arange(30)*8)
    indicator = torch.ones(240)
    indicator[pgm_idx] = 0
    params_idx = torch.arange(240)[indicator == 1]

    return (pg_codes[:, pgm_idx]- 57, pg_codes[:, params_idx] - 27)

class ShapeCompiler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        vae,
        num_text_tokens = 10000,
        text_seq_len = 256,
        depth,
        heads = 8,
        dim_head = 64,
        reversible = False,
        attn_dropout = 0.,
        ff_dropout = 0,
        sparse_attn = False,
        attn_types = None,
        loss_img_weight = 7,
        stable = False,
        sandwich_norm = False,
        shift_tokens = True,
        rotary_emb = True,
        shared_attn_ids = None,
        shared_ff_ids = None,
        share_input_output_emb = False,
        optimize_for_inference = False,
        inverse = False,
    ):
        super().__init__()
        # assert isinstance(vae, (DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE)), 'vae must be an instance of DiscreteVAE'

        num_pts_tokens = vae.num_tokens
        pts_size = vae.pts_size
        pts_fmap_size = (vae.pts_size // (2 ** vae.num_layers))
        pts_seq_len = vae.final_points

        num_text_tokens = num_text_tokens + text_seq_len  # reserve unique padding tokens for each position (text seq len)

        pg_seq_len = 240
        self.text_pos_emb = nn.Embedding(text_seq_len + 1, dim) if not rotary_emb else always(0) # +1 for <bos>
        self.text_pos_emb2 = nn.Embedding(text_seq_len, dim) if not rotary_emb else always(0) # +1 for <bos>
        # self.pts_intext_pos_emb = nn.Embedding(pts_seq_len, dim) if not rotary_emb else always(0)
        self.pg_pos_emb = nn.Embedding(pg_seq_len, dim) if not rotary_emb else always(0)
        self.pts_pos_emb = nn.Embedding(pts_seq_len + 1, dim) if not rotary_emb else always(0)
        self.pts_pos_emb1 = nn.Embedding(pts_seq_len, dim) if not rotary_emb else always(0)
        self.pts_pos_emb2 = nn.Embedding(pts_seq_len + 1, dim) if not rotary_emb else always(0)

        self.pts_pos_emb3 = nn.Embedding(pts_seq_len + 1, dim) if not rotary_emb else always(0)
        self.pts_pos_emb4 = nn.Embedding(pts_seq_len, dim) if not rotary_emb else always(0)
        # self.pts_pos_emb = AxialPositionalEmbedding(dim, axial_shape = (pts_fmap_size, pts_fmap_size)) if not rotary_emb else always(0)
        # self.pts_pos_emb = nn.Embedding(pts_seq_len, dim) if not rotary_emb else always(0)
        # AxialPositionalEmbedding(dim, axial_shape = (pts_fmap_size, pts_fmap_size)) if not rotary_emb else always(0)

        self.num_text_tokens = num_text_tokens # for offsetting logits index and calculating cross entropy loss
        self.num_pts_tokens = num_pts_tokens
        num_pg_tokens = 78
        self.num_pg_tokens = num_pg_tokens

        self.text_seq_len = text_seq_len
        self.pts_seq_len = pts_seq_len

        seq_len = text_seq_len + pts_seq_len
        # 50176 = 49664 + 512
        self.total_seq_len = seq_len

        self.pg_seq_len = pg_seq_len
        self.total_pg_seq_len = pts_seq_len + pg_seq_len
        total_tokens = num_text_tokens + num_pts_tokens
        self.total_tokens = total_tokens
        total_pg_tokens = num_pg_tokens + num_pts_tokens
        self.total_pg_tokens = total_pg_tokens

        self.vae = vae
        set_requires_grad(self.vae, False) # freeze VAE from being trained

        self.transformer = Transformer(
            dim = dim,
            causal = True,
            # seq_len = seq_len,
            seq_len = self.total_pg_seq_len,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            reversible = reversible,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            attn_types = attn_types,
            image_fmap_size = pts_fmap_size,
            sparse_attn = sparse_attn,
            stable = stable,
            sandwich_norm = sandwich_norm,
            shift_tokens = shift_tokens,
            rotary_emb = rotary_emb,
            shared_attn_ids = shared_attn_ids,
            shared_ff_ids = shared_ff_ids,
            optimize_for_inference = optimize_for_inference,
        )

        self.stable = stable

        if stable:
            self.norm_by_max = DivideMax(dim = -1)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.total_tokens),
        )

        self.to_logits_inverse = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.total_tokens),
        )

        self.to_pg_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.total_pg_tokens),
        )

        if share_input_output_emb:
            self.text_emb = SharedEmbedding(self.to_logits[1], 0, num_text_tokens)
            self.pts_emb = SharedEmbedding(self.to_logits[1], num_text_tokens, total_tokens)
        else:
            self.text_emb = nn.Embedding(num_text_tokens, dim)
            self.pts_emb = nn.Embedding(num_pts_tokens, dim)
            self.pgm_emb = nn.Embedding(num_pg_tokens, dim)

        seq_range = torch.arange(seq_len)
        logits_range = torch.arange(total_tokens)

        seq_range = rearrange(seq_range, 'n -> () n ()')
        logits_range = rearrange(logits_range, 'd -> () () d')

        logits_mask = (
            ((seq_range >= text_seq_len) & (logits_range < num_text_tokens)) |
            ((seq_range < text_seq_len) & (logits_range >= num_text_tokens))
        )
        self.register_buffer('logits_mask', logits_mask, persistent=False)

        logits_mask_inverse = (
            ((seq_range >= pts_seq_len) & (logits_range < num_pts_tokens)) |
            ((seq_range < pts_seq_len) & (logits_range >= num_pts_tokens))
        )
        self.register_buffer('logits_mask_inverse', logits_mask_inverse, persistent=False)

        pgpts_seq_len = pts_seq_len + pg_seq_len
        pg_seq_range = torch.arange(pgpts_seq_len)
        pg_logits_range = torch.arange(total_pg_tokens)
        pg_seq_range = rearrange(pg_seq_range, 'n -> () n ()')
        pg_logits_range = rearrange(pg_logits_range, 'd -> () () d')
        pg_logits_mask = (
            ((pg_seq_range >= pts_seq_len) & (pg_logits_range < num_pts_tokens)) |
            ((pg_seq_range < pts_seq_len) & (pg_logits_range >= num_pts_tokens))
        )
        self.register_buffer('pg_logits_mask', pg_logits_mask, persistent=False)

        pg_logits_mask_inverse = (
            ((pg_seq_range >= pg_seq_len) & (pg_logits_range < num_pg_tokens)) |
            ((pg_seq_range < pg_seq_len) & (pg_logits_range >= num_pg_tokens))
        )
        self.register_buffer('pg_logits_mask_inverse', pg_logits_mask_inverse, persistent=False)

        completion_seq_len = pts_seq_len + pts_seq_len
        self.total_completion_seq_len = completion_seq_len
        total_completion_tokens = num_pts_tokens + num_pts_tokens
        self.total_completion_tokens = total_completion_tokens
        self.to_completion_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.total_completion_tokens),
        )
        completion_seq_range = torch.arange(completion_seq_len)
        completion_logits_range = torch.arange(total_completion_tokens)
        completion_seq_range = rearrange(completion_seq_range, 'n -> () n ()')
        completion_logits_range = rearrange(completion_logits_range, 'd -> () () d')
        completion_logits_mask = (
            ((completion_seq_range >= pts_seq_len) & (completion_logits_range < num_pts_tokens)) |
            ((completion_seq_range < pts_seq_len) & (completion_logits_range >= num_pts_tokens))
        )
        self.register_buffer('completion_logits_mask', completion_logits_mask, persistent=False)

        self.loss_img_weight = loss_img_weight


    @torch.no_grad()
    @eval_decorator
    def generate_texts(
        self,
        tokenizer,
        text = None,
        *,
        filter_thres = 0.5,
        temperature = 1.
    ):
        text_seq_len = self.text_seq_len
        if text is None or text == "":
            text_tokens = torch.tensor([[0]]).cuda()
        else:
            text_tokens = torch.tensor(tokenizer.encode(text)).cuda().unsqueeze(0)

        for _ in range(text_tokens.shape[1], text_seq_len):
            device = text_tokens.device

            tokens = self.text_emb(text_tokens)
            tokens += self.text_pos_emb(torch.arange(text_tokens.shape[1], device = device))

            seq_len = tokens.shape[1]

            output_transf = self.transformer(tokens)

            if self.stable:
                output_transf = self.norm_by_max(output_transf)

            logits = self.to_logits(output_transf)

            # mask logits to make sure text predicts text (except last token), and pts predicts pts

            logits_mask = self.logits_mask[:, :seq_len]
            max_neg_value = -torch.finfo(logits.dtype).max
            logits.masked_fill_(logits_mask, max_neg_value)
            logits = logits[:, -1, :]

            filtered_logits = top_k(logits, thres = filter_thres)
            sample = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            text_tokens = torch.cat((text_tokens, sample[:, None]), dim=-1)

        padding_tokens = set(np.arange(self.text_seq_len) + (self.num_text_tokens - self.text_seq_len))
        texts = [tokenizer.decode(text_token, pad_tokens=padding_tokens) for text_token in text_tokens]
        return text_tokens, texts

    @torch.no_grad()
    @eval_decorator
    def generate_pts_condtext(
        self,
        text,
        *,
        clip = None,
        filter_thres = 0.5,
        temperature = 1.,
        img = None,
        num_init_img_tokens = None,
        cond_scale = 1.,
        use_cache = False,
    ):
        vae, text_seq_len, pts_seq_len, num_text_tokens = self.vae, self.text_seq_len, self.pts_seq_len, self.num_text_tokens
        total_len = text_seq_len + pts_seq_len

        text = text[:, :text_seq_len] # make sure text is within bounds
        out = text

        if exists(img):
            assert img.shape[1] == 3 and img.shape[2] == pts_size and img.shape[3] == pts_size, f'input pts must have the correct pts size {pts_size}'

            indices = vae.get_codebook_indices(img)
            num_img_tokens = default(num_init_img_tokens, int(0.4375 * pts_seq_len))  # OpenAI used 14 * 32 initial tokens to prime
            assert num_img_tokens < pts_seq_len, 'number of initial pts tokens for priming must be less than the total pts token sequence length'

            indices = indices[:, :num_img_tokens]
            out = torch.cat((out, indices), dim = -1)

        prev_cache = None
        cache = {} if use_cache else None
        for cur_len in range(out.shape[1], total_len):
            is_pts = cur_len >= text_seq_len

            text, pts = out[:, :text_seq_len], out[:, text_seq_len:]
            # print(cur_len, text.shape, pts.shape)

            if cond_scale != 1 and use_cache:
                # copy the cache state to infer from the same place twice
                prev_cache = cache.copy()

            logits = self(text, pts, cache = cache)

            if cond_scale != 1:
                # discovery by Katherine Crowson
                # https://twitter.com/RiversHaveWings/status/1478093658716966912
                null_cond_logits = self(text, pts, null_cond_prob = 1., cache = prev_cache)
                logits = null_cond_logits + (logits - null_cond_logits) * cond_scale

            logits = logits[:, -1, :]

            filtered_logits = top_k(logits, thres = filter_thres)
            sample = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            sample -= (num_text_tokens if is_pts else 0) # offset sampled token if it is an pts token, since logit space is composed of text and then pts tokens
            out = torch.cat((out, sample[:, None]), dim=-1)

        text_seq = out[:, :text_seq_len]

        pts_seq = out[:, -pts_seq_len:]
        pts = vae.decode(pts_seq)

        if exists(clip):
            scores = clip(text_seq, pts, return_loss = False)
            return pts, scores

        return pts

    def forward(
        self,
        text = None,
        pts = None,
        pg_data = None,
        pp_data = None,
        return_loss = False,
        null_cond_prob = 0.,
        cache = None,
        pg_train = True,
        pg_infer = False,
        inverse = False,
        fixed_pos = False,
        discrete_type = 1,
        pgm_only = False,
        completion_only = False,
        do_completion = True,
        writer = None,
        global_step = None,
        textshape_only = False,
        shapetext_only = False,
    ):
        debug = False
        if debug:
            from pytorch3d.io import save_ply
            import os
            save_dir = './shape2prog/vqprogram_outputs/test100'
            pts_save_dir = './shape2prog/vqprogram_outputs/test100/pts'
            pg_pts, pgms, pgms_masks, params, params_masks = pg_data[0].cuda(), pg_data[1].cuda(), pg_data[2].cuda(), pg_data[3].cuda(), pg_data[4].cuda()
            save_obj = {
                'pgm': pgms,
                'param': params,
            }
            torch.save(save_obj, os.path.join(save_dir,'%04d'%(0)+'.pt'))
            pc = pg_pts
            for i in range(pc.shape[0]):
                save_ply(os.path.join(pts_save_dir,'%04d'%i+'_ori.ply'), pc[i])

        if inverse:
            text2 = text.clone()
            pts2 = pts.clone()
        if (not pg_infer and not pgm_only and not completion_only and not shapetext_only) or textshape_only:
            assert text.shape[-1] == self.text_seq_len, f'the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})'
            batch, device, total_seq_len = text.shape[0], text.device, self.total_seq_len

            # randomly remove text condition with <null_cond_prob> probability

            if null_cond_prob > 0:
                null_mask = prob_mask_like((batch,), null_cond_prob, device = device)
                text *= rearrange(~null_mask, 'b -> b 1')

            # make sure padding in text tokens get unique padding token id

            
            text_range = torch.arange(self.text_seq_len, device = device) + (self.num_text_tokens - self.text_seq_len)
            text = torch.where(text == 0, text_range, text)

            # add <bos>

            text = F.pad(text, (1, 0), value = 0)

            tokens = self.text_emb(text)
            tokens += self.text_pos_emb(torch.arange(text.shape[1], device = device))

            seq_len = tokens.shape[1]

            if exists(pts) and not is_empty(pts):
                is_raw_pts = len(pts.shape) == 3

                if is_raw_pts:
                    pts = self.vae.get_codebook_indices(pts)

                pts_len = pts.shape[1]
                pts_emb = self.pts_emb(pts)

                pts_emb += self.pts_pos_emb1(torch.arange(pts_emb.shape[1], device = device))

                tokens = torch.cat((tokens, pts_emb), dim = 1)

                seq_len += pts_len

            # when training, if the length exceeds the total text + pts length
            # remove the last token, since it needs not to be trained

            if tokens.shape[1] > total_seq_len:
                seq_len -= 1
                tokens = tokens[:, :-1]

            if self.stable:
                alpha = 0.1
                tokens = tokens * alpha + tokens.detach() * (1 - alpha)

            if exists(cache) and cache.get('offset'):
                tokens = tokens[:, -1:]

            #tokens.shape: [24, 320, 512]
            out = self.transformer(tokens, cache=cache)
            #out [24, 320, 512]

            if self.stable:
                out = self.norm_by_max(out)

            # out.shape: [4, 273, 512]
            # logits.shape: [4, 273, 50176]
            logits = self.to_logits(out)

            # mask logits to make sure text predicts text (except last token), and pts predicts pts

            logits_mask = self.logits_mask[:, :seq_len]
            if exists(cache) and cache.get('offset'):
                logits_mask = logits_mask[:, -1:]
            max_neg_value = -torch.finfo(logits.dtype).max
            logits.masked_fill_(logits_mask, max_neg_value)

            if exists(cache):
                cache['offset'] = cache.get('offset', 0) + logits.shape[1]

            if not return_loss:
                return logits

            assert exists(pts), 'when training, pts must be supplied'

            offsetted_pts = pts + self.num_text_tokens
            labels = torch.cat((text[:, 1:], offsetted_pts), dim = 1)

            logits = rearrange(logits, 'b n c -> b c n')

            loss_text = F.cross_entropy(logits[:, :, :self.text_seq_len], labels[:, :self.text_seq_len])
            loss_img = F.cross_entropy(logits[:, :, self.text_seq_len:], labels[:, self.text_seq_len:])

            # loss = (loss_text + self.loss_img_weight * loss_img) / (self.loss_img_weight + 1)
            # loss = loss_text + loss_img
            loss = loss_text + self.loss_img_weight * loss_img
            if textshape_only:
                print('step: %d, total_loss:%.3f, loss_text:%.3f, loss_img:%.3f, '%(global_step, loss, loss_text, loss_img))

            if writer is not None:
                writer.add_scalar('text2pts_loss_text', loss_text.item(), global_step)
                writer.add_scalar('text2pts_loss_img', loss_img.item(), global_step)

        if ((pg_train and not pg_infer) or pgm_only) and not completion_only and not textshape_only and not shapetext_only:
            pg_pts, pgms, pgms_masks, params, params_masks = pg_data[0].cuda(), pg_data[1].cuda(), pg_data[2].cuda(), pg_data[3].cuda(), pg_data[4].cuda()
            device = pg_pts.device
            
            pts_code = self.vae.get_codebook_indices(pg_pts)
            # why?
            # pts_range = torch.arange(self.pts_seq_len, device = device) + (self.num_pts_tokens - self.pts_seq_len)
            # pts_code = torch.where(pts_code == 0, pts_range, pts_code)

            pts_code = F.pad(pts_code, (1, 0), value = 0)
            pts_emb = self.pts_emb(pts_code)
            pts_emb += self.pts_pos_emb2(torch.arange(pts_emb.shape[1], device = device))
            pg_seq_len = pts_code.shape[1]

            # pg_code = self.pgvae.get_codebook_indices(pgms, params)
            if discrete_type == 1:
                pg_code = map_pgcode(pgms, params)
            elif discrete_type == 2:
                pg_code, pgm_idx, param_idx = map_pgcode2(pgms, params)
            else:
                assert NameError('non-exist discrete way')

            #debug
            # out = self.pgvae.decode(pg_code)
            # id = 2
            # pgms_our = torch.argmax(out[id,:,:22].reshape(10,3,-1),-1)
            # pgms_in = pgms[id,:]
            
            pg_emb = self.pgm_emb(pg_code)
            pg_emb += self.pg_pos_emb(torch.arange(pg_emb.shape[1], device = device))
            pg_tokens = torch.cat((pts_emb, pg_emb), dim = 1)
            pg_seq_len += pg_emb.shape[1]
            
            if pg_seq_len > self.total_pg_seq_len:
                pg_seq_len -= 1
                pg_tokens = pg_tokens[:, :-1]

            pg_out = self.transformer(pg_tokens)
            # pg_out = self.transformer2(pg_tokens)
            pg_logits = self.to_pg_logits(pg_out)
            
            pg_logits_mask = self.pg_logits_mask[:, :pg_seq_len]
            max_neg_value = -torch.finfo(pg_logits.dtype).max
            pg_logits.masked_fill_(pg_logits_mask, max_neg_value)

            if not fixed_pos:
                if discrete_type == 1:
                    offsetted_pg_code = pg_code + self.num_pts_tokens
                    pg_labels = torch.cat((pts_code[:, 1:], offsetted_pg_code), dim = 1)
                    pg_logits = rearrange(pg_logits, 'b n c -> b c n')

                    loss_pts = F.cross_entropy(pg_logits[:, :, :self.pts_seq_len], pg_labels[:, :self.pts_seq_len])
                    loss_pg = F.cross_entropy(pg_logits[:, :, self.pts_seq_len:], pg_labels[:, self.pts_seq_len:])
                elif discrete_type == 2:
                    offsetted_pg_code = pg_code + self.num_pts_tokens
                    pg_labels = torch.cat((pts_code[:, 1:], offsetted_pg_code), dim = 1)
                    pg_logits = rearrange(pg_logits, 'b n c -> b c n')
                    loss_pts = F.cross_entropy(pg_logits[:, :, :self.pts_seq_len], pg_labels[:, :self.pts_seq_len])

                    pgms = pg_logits[:,512:512+21,self.pts_seq_len:][:,:,pgm_idx]
                    pred = F.log_softmax(pgms, dim=1)
                    target = pg_code[:, pgm_idx] - 57
                    pgms_masks = pgms_masks.contiguous().view(pgms.shape[0], -1)
                    loss_cls = - pred.gather(1, target.unsqueeze(1)) * pgms_masks.reshape(pgms.shape[0],1,-1)
                    loss_pg1 = torch.sum(loss_cls) / torch.sum(pgms_masks)

                    params = pg_logits[:,512+21:,self.pts_seq_len:][:,:,param_idx]
                    pred = F.log_softmax(params, dim=1)
                    target = pg_code[:, param_idx]
                    params_masks = params_masks.contiguous().view(pgms.shape[0], -1)
                    loss_reg = - pred.gather(1, target.unsqueeze(1)) * params_masks.reshape(pgms.shape[0],1,-1)
                    loss_pg2 = torch.sum(loss_reg) / torch.sum(params_masks)

                    # loss_pg1 = F.cross_entropy(pg_logits[:,512:512+21,self.pts_seq_len:][:,:,pgm_idx], pg_code[:, pgm_idx])
                    # loss_pg2 = F.cross_entropy(pg_logits[:,512+21:,self.pts_seq_len:][:,:,param_idx], pg_code[:, param_idx])
                    loss_pg = loss_pg1 + 0.2*loss_pg2
                    # loss_pg = (loss_pg1 + 5*loss_pg2)/6

                else:
                    NameError('non-exist type')

            else:
                neg_filled_value = torch.ones(pg_logits.shape[0], pg_logits.shape[1], self.num_text_tokens).cuda()
                filled_logits = torch.cat((neg_filled_value,pg_logits),-1)
                offsetted_pg_code = pg_code + self.num_pts_tokens + self.num_text_tokens
                pg_labels = torch.cat((pts_code[:, 1:] + self.num_text_tokens, offsetted_pg_code), dim = 1)

                pg_logits = rearrange(filled_logits, 'b n c -> b c n')

                loss_pts = F.cross_entropy(pg_logits[:, :, :self.pts_seq_len], pg_labels[:, :self.pts_seq_len])
                loss_pg = F.cross_entropy(pg_logits[:, :, self.pts_seq_len:], pg_labels[:, self.pts_seq_len:])


            # loss = 1/2*loss + 1/2*((loss_pg + self.loss_img_weight *loss_pts)/ (self.loss_img_weight +1))
            # loss = (loss_pg + self.loss_img_weight *loss_pts)/ (self.loss_img_weight +1)
            # loss = loss_pg + loss_pts
            if pgm_only:
                loss = loss_pg + loss_pts
            else:
                # loss = loss + loss_pg + loss_pts
                # loss = loss + loss_pg + self.loss_img_weight * loss_pts
                loss = loss + loss_pts + self.loss_img_weight * loss_pg
            # loss = (loss_pg + loss_pts)/ (1+1)
            if pgm_only:
                print('total_loss:%.3f, loss_pg:%.3f, loss_pg1:%.3f, loss_pg2:%.3f, loss_pts:%.3f'%(loss, loss_pg, loss_pg1, loss_pg2, loss_pts))
            else:
                if discrete_type == 1:
                    print('total_loss:%.3f, loss_text:%.3f, loss_img:%d x %.3f, loss_pg:%.3f, loss_pts:%.3f'%(loss, loss_text, self.loss_img_weight, loss_img, loss_pg, loss_pts))
                else:
                    print('step: %d, total_loss:%.3f, loss_text:%.3f, loss_img:%d x %.3f, loss_pg:%.3f, loss_pg1:%.3f, loss_pg2:%.3f, loss_pts:%.3f'%(global_step, loss, loss_text, self.loss_img_weight, loss_img, loss_pg, loss_pg1, loss_pg2, loss_pts))
            if writer is not None:
                writer.add_scalar('pts2pgm_loss_pg', loss_pg.item(), global_step)
                writer.add_scalar('pts2pgm_loss_pg1', loss_pg1.item(), global_step)
                writer.add_scalar('pts2pgm_loss_pg2', loss_pg2.item(), global_step)
                writer.add_scalar('pts2pgm_loss_pts', loss_pts.item(), global_step)

        if do_completion and not pgm_only and not textshape_only and not shapetext_only:
            inputs = pp_data[0].cuda()
            targets = pp_data[1].cuda()
            device = inputs.device
            pts_code = self.vae.get_codebook_indices(inputs)
            pts_code = F.pad(pts_code, (1, 0), value = 0)
            pts_emb = self.pts_emb(pts_code)
            pts_emb += self.pts_pos_emb3(torch.arange(pts_emb.shape[1], device = device))
            completion_seq_len = pts_code.shape[1]

            pts_code_target = self.vae.get_codebook_indices(targets)
            pts_emb_target = self.pts_emb(pts_code_target)
            pts_emb_target += self.pts_pos_emb4(torch.arange(pts_emb_target.shape[1], device = device))
            completion_tokens = torch.cat((pts_emb, pts_emb_target), dim = 1)
            completion_seq_len += pts_emb_target.shape[1]
            if completion_seq_len > self.total_completion_seq_len:
                completion_seq_len -= 1
                completion_tokens = completion_tokens[:, :-1]
            completion_out = self.transformer(completion_tokens)
            # pg_out = self.transformer2(pg_tokens)
            completion_logits = self.to_completion_logits(completion_out)

            completion_logits_mask = self.completion_logits_mask[:, :completion_seq_len]
            max_neg_value = -torch.finfo(completion_logits.dtype).max
            completion_logits.masked_fill_(completion_logits_mask, max_neg_value)

            offsetted_completion_target = pts_code_target + self.num_pts_tokens
            completion_labels = torch.cat((pts_code[:, 1:], offsetted_completion_target), dim = 1)
            completion_logits = rearrange(completion_logits, 'b n c -> b c n')
            loss_input = F.cross_entropy(completion_logits[:, :, :self.pts_seq_len], completion_labels[:, :self.pts_seq_len])
            loss_target = F.cross_entropy(completion_logits[:, :, self.pts_seq_len:], completion_labels[:, self.pts_seq_len:])
            if completion_only:
                loss = loss_input + self.loss_img_weight * loss_target
            else:
                # loss = loss + loss_input + loss_target
                loss = loss + loss_input + self.loss_img_weight * loss_target
            print('step: %d, completion_loss:%.3f, loss_input:%.3f, weight:%d * loss_target:%.3f'%(global_step, loss, loss_input, self.loss_img_weight, loss_target))
            if writer is not None:
                writer.add_scalar('pts2pts_loss_input', loss_input.item(), global_step)
                writer.add_scalar('pts2pts_loss_target', loss_target.item(), global_step)

        if inverse:
            text = text2
            pts = pts2
            batch, device, total_seq_len = text.shape[0], text.device, self.total_seq_len

            # randomly remove text condition with <null_cond_prob> probability

            if null_cond_prob > 0:
                null_mask = prob_mask_like((batch,), null_cond_prob, device = device)
                text *= rearrange(~null_mask, 'b -> b 1')

            # make sure padding in text tokens get unique padding token id

            text_range = torch.arange(self.text_seq_len, device = device) + (self.num_text_tokens - self.text_seq_len)
            text = torch.where(text == 0, text_range, text)

            pts = self.vae.get_codebook_indices(pts)
            pts = F.pad(pts, (1, 0), value = 0)
            pts_emb = self.pts_emb(pts)
            pts_emb += self.pts_pos_emb(torch.arange(pts_emb.shape[1], device = device))

            tokens = self.text_emb(text)
            tokens += self.text_pos_emb2(torch.arange(text.shape[1], device = device))
            tokens = torch.cat((pts_emb, tokens), dim = 1)
            seq_len = tokens.shape[1]
            if tokens.shape[1] > total_seq_len:
                seq_len -= 1
                tokens = tokens[:, :-1]
            out = self.transformer(tokens, cache=cache)
            logits = self.to_logits_inverse(out)

            logits_mask = self.logits_mask_inverse[:, :seq_len]
            if exists(cache) and cache.get('offset'):
                logits_mask = logits_mask[:, -1:]
            max_neg_value = -torch.finfo(logits.dtype).max
            logits.masked_fill_(logits_mask, max_neg_value)
            
            offsetted_text = text + self.num_pts_tokens
            labels = torch.cat((pts[:, 1:], offsetted_text), dim = 1)
            logits = rearrange(logits, 'b n c -> b c n')

            loss_img2 = F.cross_entropy(logits[:, :, :self.pts_seq_len], labels[:, :self.pts_seq_len])
            loss_text2 = F.cross_entropy(logits[:, :, self.pts_seq_len:], labels[:, self.pts_seq_len:])

            if shapetext_only:
                loss = loss_img2 + loss_text2
            else:
                # loss = loss + loss_img2 + loss_text2
                loss = loss + loss_img2 + self.loss_img_weight*loss_text2
            print('step:%d, total_loss2:%.3f, loss_text2:%.3f, loss_img2:%d x %.3f, '%(global_step, loss, loss_text2, 1, loss_img2))

            if writer is not None:
                writer.add_scalar('pts2text_loss_pts', loss_img2.item(), global_step)
                writer.add_scalar('pts2text_loss_text', loss_text2.item(), global_step)


            if False:

                ## loss 1.
            
                pg_pts, pgms, _, params, _ = pg_data[0].cuda(), pg_data[1].cuda(), pg_data[2].cuda(), pg_data[3].cuda(), pg_data[4].cuda()
            
                pts_code = self.vae.get_codebook_indices(pg_pts)
                pts_emb = self.pts_emb(pts_code)
                pg_seq_len = pts_code.shape[1]

                # pg_code = self.pgvae.get_codebook_indices(pgms, params)
                pg_code = map_pgcode(pgms, params)
                pg_code = F.pad(pg_code, (1, 0), value = 0)
            
                #debug
                # out = self.pgvae.decode(pg_code)
                # id = 2
                # pgms_our = torch.argmax(out[id,:,:22].reshape(10,3,-1),-1)
                # pgms_in = pgms[id,:]
            
                pg_emb = self.pgm_emb(pg_code)
                pg_tokens = torch.cat((pg_emb, pts_emb), dim = 1)
                pg_seq_len += pg_emb.shape[1]
            
                if pg_seq_len > self.total_pg_seq_len:
                    pg_seq_len -= 1
                    pg_tokens = pg_tokens[:, :-1]

                pg_out = self.transformer(pg_tokens)
                # pg_out = self.transformer2(pg_tokens)
                pg_logits = self.to_pg_logits(pg_out)
            
                pg_logits_mask = self.pg_logits_mask_inverse[:, :pg_seq_len]
                max_neg_value = -torch.finfo(pg_logits.dtype).max
                pg_logits.masked_fill_(pg_logits_mask, max_neg_value)

                offsetted_pts_code = pts_code + self.num_pg_tokens
                pg_labels = torch.cat((pg_code[:, 1:], offsetted_pts_code), dim = 1)

                pg_logits = rearrange(pg_logits, 'b n c -> b c n')

                loss_pg2 = F.cross_entropy(pg_logits[:, :, :self.pg_seq_len], pg_labels[:, :self.pg_seq_len])
                loss_pts2 = F.cross_entropy(pg_logits[:, :, self.pg_seq_len:], pg_labels[:, self.pg_seq_len:])


                loss = 1/2*loss + 1/4*(loss_pg2 + loss_pts2 + loss_img2 + loss_text2)
                print('total_loss2:%.3f, loss_text2:%.3f, loss_img2:%d x %.3f, loss_pg2:%.3f, loss_pts2:%.3f'%(loss, loss_text2, 1, loss_img2, loss_pg2, loss_pts2))



        return loss

    @torch.no_grad()
    @eval_decorator
    def generate_pgm_condpts(
        self,
        pg_pts,
        pg = None,
        *,
        filter_thres = 0.5,
        temperature = 1.
    ):
        pts_seq_len, pg_seq_len, num_pts_tokens = self.pts_seq_len, self.pg_seq_len, self.num_pts_tokens
        total_len = pts_seq_len + pg_seq_len
        
        pts_code = self.vae.get_codebook_indices(pg_pts)
        out = pts_code
        pgm_idx = (torch.arange(30)*8)
        indicator = torch.ones(240)
        indicator[pgm_idx] = 0
        params_idx = torch.arange(240)[indicator == 1]

        for cur_len in range(out.shape[1], total_len):
            # is_pg = cur_len >= pts_seq_len

            pts_code, pg_code = out[:, :pts_seq_len], out[:, pts_seq_len:]
            
            # pts_range = torch.arange(self.pts_seq_len, device = pg_pts.device) + (self.num_pts_tokens - self.pts_seq_len)
            # pts_code = torch.where(pts_code == 0, pts_range, pts_code)
            pts_code = F.pad(pts_code, (1, 0), value = 0)
            pts_emb = self.pts_emb(pts_code)
            pts_emb += self.pts_pos_emb2(torch.arange(pts_emb.shape[1], device = pts_emb.device))
            # pts_emb += self.pts_pos_emb(torch.arange(pts_emb.shape[1], device = pts_emb.device))
            pg_emb = self.pgm_emb(pg_code)
            pg_emb += self.pg_pos_emb(torch.arange(pg_emb.shape[1], device = pts_emb.device))

            ptspg_emb = torch.cat((pts_emb, pg_emb), dim = 1)
            cur_pg_seq_len = ptspg_emb.shape[1]
            if ptspg_emb.shape[1] > total_len:
                cur_pg_seq_len -= 1
                ptspg_emb = ptspg_emb[:, :-1]
            
            cur_out = self.transformer(ptspg_emb)
            cur_logits = self.to_pg_logits(cur_out)
            pg_logits_mask = self.pg_logits_mask[:, :cur_pg_seq_len]
            max_neg_value = -torch.finfo(cur_logits.dtype).max
            cur_logits.masked_fill_(pg_logits_mask, max_neg_value)

            logits = cur_logits[:, -1, :]

            filtered_logits = top_k(logits, thres = 0.9)
            if cur_len - 128 in pgm_idx:
                sample = gumbel_sample(filtered_logits[:, 512:512+21], temperature = temperature, dim = -1) + 57
                # sample = gumbel_sample(filtered_logits[:, 512:512+21], temperature = temperature, dim = -1)
            else:
                sample = gumbel_sample(filtered_logits[:, 512+21:], temperature = temperature, dim = -1)

            # sample = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            # sample -= (num_pts_tokens if is_pg else 0) # offset sampled token if it is an pts token, since logit space is composed of text and then pts tokens
            out = torch.cat((out, sample[:, None]), dim=-1)
        
        pg_seq = out[:, -pg_seq_len:]
        return decode_pgcode2(pg_seq)

    @torch.no_grad()
    @eval_decorator
    def generate_text_condpts(
        self,
        tokenizer,
        pg_pts,
        pg = None,
        *,
        filter_thres = 0.5,
        temperature = 1.
    ):
        pts_seq_len, text_seq_len, num_pts_tokens = self.pts_seq_len, self.text_seq_len, self.num_pts_tokens
        total_len = pts_seq_len + text_seq_len
        
        pts_code = self.vae.get_codebook_indices(pg_pts)
        out = pts_code

        for cur_len in range(out.shape[1], total_len):
            # is_pg = cur_len >= pts_seq_len

            pts_code, text_code = out[:, :pts_seq_len], out[:, pts_seq_len:]
            
            # pts_range = torch.arange(self.pts_seq_len, device = pg_pts.device) + (self.num_pts_tokens - self.pts_seq_len)
            # pts_code = torch.where(pts_code == 0, pts_range, pts_code)
            pts_code = F.pad(pts_code, (1, 0), value = 0)
            pts_emb = self.pts_emb(pts_code)
            pts_emb += self.pts_pos_emb(torch.arange(pts_emb.shape[1], device = pts_emb.device))
            # pts_emb += self.pts_pos_emb(torch.arange(pts_emb.shape[1], device = pts_emb.device))
            text_emb = self.text_emb(text_code)
            text_emb += self.text_pos_emb2(torch.arange(text_emb.shape[1], device = pts_emb.device))

            ptstext_emb = torch.cat((pts_emb, text_emb), dim = 1)
            cur_ptstext_seq_len = ptstext_emb.shape[1]
            if ptstext_emb.shape[1] > total_len:
                cur_ptstext_seq_len -= 1
                ptstext_emb = ptstext_emb[:, :-1]
            
            cur_out = self.transformer(ptstext_emb)
            cur_logits = self.to_logits_inverse(cur_out)
            # cur_logits = self.to_logits(cur_out)
            ptstext_logits_mask = self.logits_mask_inverse[:, :cur_ptstext_seq_len]
            max_neg_value = -torch.finfo(cur_logits.dtype).max
            cur_logits.masked_fill_(ptstext_logits_mask, max_neg_value)

            logits = cur_logits[:, -1, :]

            filtered_logits = top_k(logits, thres = 0.9)
            sample = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            sample -= num_pts_tokens
            out = torch.cat((out, sample[:, None]), dim=-1)
        
        text_tokens = out[:, pts_seq_len:]
        padding_tokens = set(np.arange(self.text_seq_len) + (self.num_text_tokens - self.text_seq_len))
        # texts = [tokenizer.tokenizer.decode(text_token, pad_tokens=padding_tokens) for text_token in text_tokens]
        texts = [tokenizer.decode(text_token, pad_tokens=padding_tokens) for text_token in text_tokens]
        return text_tokens, texts


    @torch.no_grad()
    @eval_decorator
    def generate_pts_cond_pts(
        self,
        pg_pts,
        pg = None,
        *,
        filter_thres = 0.5,
        temperature = 1.
    ):
        pts_seq_len, text_seq_len, num_pts_tokens = self.pts_seq_len, self.text_seq_len, self.num_pts_tokens
        total_len = pts_seq_len + pts_seq_len
        
        pts_code = self.vae.get_codebook_indices(pg_pts)
        out = pts_code

        for cur_len in range(out.shape[1], total_len):
            # is_pg = cur_len >= pts_seq_len

            pts_code, pts_code2 = out[:, :pts_seq_len], out[:, pts_seq_len:]
            
            # pts_range = torch.arange(self.pts_seq_len, device = pg_pts.device) + (self.num_pts_tokens - self.pts_seq_len)
            # pts_code = torch.where(pts_code == 0, pts_range, pts_code)
            pts_code = F.pad(pts_code, (1, 0), value = 0)
            pts_emb = self.pts_emb(pts_code)
            pts_emb += self.pts_pos_emb3(torch.arange(pts_emb.shape[1], device = pts_emb.device))
            # pts_emb += self.pts_pos_emb(torch.arange(pts_emb.shape[1], device = pts_emb.device))
            pts_emb2 = self.pts_emb(pts_code2)
            pts_emb2 += self.pts_pos_emb4(torch.arange(pts_emb2.shape[1], device = pts_emb.device))

            ptspts2_emb = torch.cat((pts_emb, pts_emb2), dim = 1)
            cur_ptspts2_seq_len = ptspts2_emb.shape[1]
            if ptspts2_emb.shape[1] > total_len:
                cur_ptspts2_seq_len -= 1
                ptspts2_emb = ptspts2_emb[:, :-1]
            
            cur_out = self.transformer(ptspts2_emb)
            cur_logits = self.to_completion_logits(cur_out)
            # cur_logits = self.to_logits(cur_out)
            ptspts2_logits_mask = self.completion_logits_mask[:, :cur_ptspts2_seq_len]
            max_neg_value = -torch.finfo(cur_logits.dtype).max
            cur_logits.masked_fill_(ptspts2_logits_mask, max_neg_value)

            logits = cur_logits[:, -1, :]

            filtered_logits = top_k(logits, thres = 0.9)
            sample = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            sample -= num_pts_tokens
            out = torch.cat((out, sample[:, None]), dim=-1)
        
        pts2_tokens = out[:, pts_seq_len:]
        pts2 = self.vae.decode(pts2_tokens)
        # texts = [tokenizer.tokenizer.decode(text_token, pad_tokens=padding_tokens) for text_token in text_tokens]
        return pts2
