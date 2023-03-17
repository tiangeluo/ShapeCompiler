from pathlib import Path
from random import randint, choice

import PIL

from torch.utils.data import Dataset
from torchvision import transforms as T

from pytorch3d.io import load_ply
from IPython import embed
import h5py, pickle
import numpy as np
import os
import torch

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

class TextImageDataset(Dataset):
    def __init__(self,
                 folder,
                 text_len=256,
                 image_size=128,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        path = Path(folder)

        #text_files = [*path.glob('**/*.txt')]
        #image_files = [
        #    *path.glob('**/*.png'), *path.glob('**/*.jpg'),
        #    *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        #]
        text_files = [*path.glob('*.txt')]
        image_files = [*path.glob('*.ply')]

        text_files = {text_file.stem: text_file for text_file in text_files}
        image_files = {image_file.stem: image_file for image_file in image_files}

        keys = (image_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.tokenizer = tokenizer
        self.resize_ratio = resize_ratio
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.keys[ind]

        text_file = self.text_files[key]
        image_file = self.image_files[key]

        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        #try:
        #    image_tensor = self.image_transform(PIL.Image.open(image_file))
        #except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
        #    print(f"An exception occurred trying to load file {image_file}.")
        #    print(f"Skipping index {ind}")
        #    return self.skip_sample(ind)

        ## Success
        #return tokenized_text, image_tensor
        pc = load_ply(image_file)
        points = normalize_points_torch(pc[0].unsqueeze(0)).squeeze()
        scale = points.new(1).uniform_(0.9, 1.05)
        points[:, 0:3] *= scale

        # Success
        return tokenized_text, points

class TextPtsDataset(Dataset):
    def __init__(self,
                 name,
                 text_len=256,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False,
                 validation=False,
                 return_descriptions=False,
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        root_dir = '/home/tiangel/datasets'
        if not validation:
            if name == 'shapeglot':
                pc_file = h5py.File(os.path.join(root_dir, 'shapeglot_pc_v3_train.h5'),'r')
                self.pc_data = np.array(pc_file['data'])
                pc_file.close()
                text_file = open(os.path.join(root_dir, 'shapeglot_text_v4_train.pkl'), 'rb')
                self.text_data = pickle.load(text_file)
                text_file.close()
            elif name == 'abo':
                pc_file = h5py.File(os.path.join(root_dir, 'abo_pc_v2_train.h5'),'r')
                self.pc_data = np.array(pc_file['data'])
                pc_file.close()
                text_file = open(os.path.join(root_dir, 'abo_text_v4_train.pkl'), 'rb')
                self.text_data = pickle.load(text_file)
                text_file.close()
            elif name == 'text2shape':
                pc_file = h5py.File(os.path.join(root_dir, 'text2shape_pc_v3_train.h5'),'r')
                self.pc_data = np.array(pc_file['data'])
                pc_file.close()
                text_file = open(os.path.join(root_dir, 'text2shape_text_v4_train.pkl'), 'rb')
                self.text_data = pickle.load(text_file)
                text_file.close()
            else:
                NameError('unsupported datasets')
        else:
            if name == 'shapeglot':
                pc_file = h5py.File(os.path.join(root_dir, 'shapeglot_pc_v3_val.h5'),'r')
                self.pc_data = np.array(pc_file['data'])
                pc_file.close()
                text_file = open(os.path.join(root_dir, 'shapeglot_text_v4_val.pkl'), 'rb')
                self.text_data = pickle.load(text_file)
                text_file.close()
            elif name == 'abo':
                pc_file = h5py.File(os.path.join(root_dir, 'abo_pc_v2_val.h5'),'r')
                self.pc_data = np.array(pc_file['data'])
                pc_file.close()
                text_file = open(os.path.join(root_dir, 'abo_text_v4_val.pkl'), 'rb')
                self.text_data = pickle.load(text_file)
                text_file.close()
            elif name == 'text2shape':
                pc_file = h5py.File(os.path.join(root_dir, 'text2shape_pc_v3_val.h5'),'r')
                self.pc_data = np.array(pc_file['data'])
                pc_file.close()
                text_file = open(os.path.join(root_dir, 'text2shape_text_v4_val.pkl'), 'rb')
                self.text_data = pickle.load(text_file)
                text_file.close()
            else:
                NameError('unsupported datasets')

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.tokenizer = tokenizer
        self.validation = validation
        self.return_descriptions = return_descriptions


    def __len__(self):
        return self.pc_data.shape[0]

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def __getitem__(self, ind):

        descriptions = self.text_data[ind]
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        
        if not self.validation:
            try:
                description = choice(descriptions)
            except IndexError as zero_captions_in_file_ex:
                print(f"Skipping index {ind}")
                return self.skip_sample(ind)

            tokenized_text = self.tokenizer.tokenize(
                description[:self.text_len],
                self.text_len,
                truncate_text=self.truncate_captions
            ).squeeze(0)
        else:
            tokenized_text = []
            for description in descriptions:
                tokenized_text.append(self.tokenizer.tokenize(
                    description[:self.text_len],
                    self.text_len,
                    truncate_text=self.truncate_captions
                ))
            tokenized_text = torch.cat(tokenized_text, dim=0)

        #try:
        #    image_tensor = self.image_transform(PIL.Image.open(image_file))
        #except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
        #    print(f"An exception occurred trying to load file {image_file}.")
        #    print(f"Skipping index {ind}")
        #    return self.skip_sample(ind)

        ## Success
        #return tokenized_text, image_tensor
        pc = torch.Tensor(self.pc_data[ind]).unsqueeze(0)
        points = normalize_points_torch(pc[0].unsqueeze(0)).squeeze()
        if not self.validation:
            scale = points.new(1).uniform_(0.9, 1.05)
            points[:, 0:3] *= scale

        # Success
        if not self.return_descriptions:
            return tokenized_text, points
        else:
            return tokenized_text, points, descriptions
