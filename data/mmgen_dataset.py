import copy
import json
import os
from typing import List

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import data.path as PathConfig


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class MMGenDataset(Dataset):
    def __init__(
        self,
        path_meta_data=PathConfig.path_meta_data,
        transoform=transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                inplace=True,
            ),
        ]),
        num_tasks=1,
        use_post_process=True,
        # rgb_ext='.JPEG',
        use_preprocess_vae=False,
        use_repa_reg=False,
        mode_load_1task=None,
        vae_h5_paths=None,
        dino_rgb_h5=PathConfig.path_h5_dino_rgb,
    ):
        super().__init__()

        if not 1 <= num_tasks <= len(PathConfig.TASK_NAMES):
            raise ValueError(
                f"num_tasks must be between 1 and {len(PathConfig.TASK_NAMES)}, got {num_tasks}"
        )

        self.mode_load_1task = mode_load_1task
        self.metas = self.prepare_metadata([path_meta_data])
        
        self.post_process = transoform
        
        self.image_format = "RGB"
        self.num_tasks = num_tasks
        self.use_post_process = use_post_process 
        if self.post_process is None:
            self.use_post_process = False
        
        self.use_repa_reg = use_repa_reg
        
        self.use_preprocess_vae = use_preprocess_vae
        if self.use_preprocess_vae:
            vae_h5_paths = vae_h5_paths or {
                'rgb': PathConfig.path_h5_rgb,
                'depth': PathConfig.path_h5_depth,
                'normal': PathConfig.path_h5_normal,
                'mask': PathConfig.path_h5_mask,
            }
            required_tasks = [PathConfig.map_idxtask_to_name(i) for i in range(num_tasks)]
            undefined = [name for name in required_tasks if name not in vae_h5_paths]
            if undefined:
                raise ValueError(
                    f"Missing VAE HDF5 path mappings for: {', '.join(undefined)}"
                )
            missing = [name for name in required_tasks if not os.path.isfile(vae_h5_paths[name])]
            if missing:
                details = ', '.join(f"{name}={vae_h5_paths[name]}" for name in missing)
                raise FileNotFoundError(f"Missing precomputed VAE HDF5 files: {details}")

            self.vae_h5_paths = vae_h5_paths
            self.h5_rgb = None
            self.h5_depth = None
            self.h5_normal = None
            self.h5_mask = None
            print('Validated precomputed VAE files')
            
        if self.use_repa_reg:
            if not os.path.isfile(dino_rgb_h5):
                raise FileNotFoundError(f"Missing RGB DINO HDF5 file: {dino_rgb_h5}")
            self.dino_rgb_h5 = dino_rgb_h5
            self.h5_dino_rgb = None


    def prepare_metadata(self, metadatas: List[str]):
        metas = []
        for metadata in metadatas:
            with open(metadata, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        metas.append(json.loads(line))
        if not metas:
            raise ValueError(f"No samples found in metadata: {metadatas}")
        print(f"Loaded {len(metas)} samples")
        return metas

    def process(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        

        if self.mode_load_1task is None:
            input_path = dataset_dict["path_rgb"]
            image = Image.open(input_path).convert('RGB')
            dataset_dict["rgb"] = image
        else:
            key = f"path_{self.mode_load_1task}"
            if self.mode_load_1task not in PathConfig.TASK_NAMES:
                raise ValueError(f"Unknown mode_load_1task: {self.mode_load_1task}")
            input_path = dataset_dict[key]
            image = Image.open(input_path).convert('RGB')
            dataset_dict["rgb"] = image
        dataset_dict["path_input"] = input_path
        
        if self.num_tasks > 1:
            depth = Image.open(dataset_dict["path_depth"]).convert('RGB')
            dataset_dict["depth"] = depth

        if self.num_tasks > 2:
            normal = Image.open(dataset_dict["path_normal"]).convert('RGB')
            dataset_dict["normal"] = normal
        
        if self.num_tasks > 3:
            mask = Image.open(dataset_dict["path_mask"]).convert('RGB')
            dataset_dict["mask"] = mask
        
        return dataset_dict

    def __len__(self):
        return len(self.metas)
    
    def _ensure_h5_open(self):
        # DataLoader workers must open their own h5py handles after forking.
        if self.use_preprocess_vae and self.h5_rgb is None:
            self.h5_rgb = h5py.File(self.vae_h5_paths['rgb'], 'r')
            if self.num_tasks > 1:
                self.h5_depth = h5py.File(self.vae_h5_paths['depth'], 'r')
            if self.num_tasks > 2:
                self.h5_normal = h5py.File(self.vae_h5_paths['normal'], 'r')
            if self.num_tasks > 3:
                self.h5_mask = h5py.File(self.vae_h5_paths['mask'], 'r')
        if self.use_repa_reg and self.h5_dino_rgb is None:
            self.h5_dino_rgb = h5py.File(self.dino_rgb_h5, 'r')

    def get_dino_feature(self, meta):
        self._ensure_h5_open()
        dino_rgb = self.h5_dino_rgb[meta['path_rel_rgb']][...]
        dino_rgb = torch.tensor(dino_rgb.copy()).float()
        return {'dino_rgb': dino_rgb, 'x_repa': dino_rgb}
     
    def get_vae_feature(self, meta):
        self._ensure_h5_open()
        rgb = self.h5_rgb[meta['path_rel_rgb']][...]
        rgb = torch.tensor(rgb.copy()).float()
        dict_output = {'rgb': rgb}
        features = [rgb]
                    
        if self.num_tasks > 1:
            depth = self.h5_depth[meta['path_rel_rgb']][...]
            depth = torch.tensor(depth.copy()).float()
            dict_output['depth'] = depth
            features.append(depth)
            
        if self.num_tasks > 2:
            normal = self.h5_normal[meta['path_rel_rgb']][...]
            normal = torch.tensor(normal.copy()).float()
            dict_output['normal'] = normal
            features.append(normal)
        
        if self.num_tasks > 3:
            mask = self.h5_mask[meta['path_rel_rgb']][...]
            mask = torch.tensor(mask.copy()).float()
            dict_output['mask'] = mask
            features.append(mask)
        
        dict_output['x'] = torch.cat(features, dim=0)
        return dict_output
            
    def __getitem__(self, index: int):
        errors = []
        for _ in range(len(self.metas)):
            index = index % len(self.metas)
            meta = self.metas[index]
            
            if self.use_preprocess_vae:
                try:
                    dict_output = {
                        'cat': meta['cat'],
                        'idx_cat': torch.tensor(meta['idx_cat']),
                        'path_rel_rgb': meta['path_rel_rgb'],
                    }
                    
                    dict_output.update(self.get_vae_feature(meta))
  
                    if self.use_repa_reg:
                        dict_output.update(self.get_dino_feature(meta))

                    return dict_output
                except Exception as e:
                    errors.append(f"{meta.get('path_rel_rgb', index)}: {e}")
                    index += 1
                    continue
            
            try:
                dataset_dict = self.process(meta)
                
                rgb = self.post_process(dataset_dict["rgb"]) if self.use_post_process else dataset_dict["rgb"]
                dict_output = {
                    'cat': dataset_dict['cat'],
                    'idx_cat': torch.tensor(dataset_dict['idx_cat']),
                    'rgb': rgb,
                    'x': rgb,
                    'path_input': dataset_dict['path_input'],
                    'path_rel_rgb': dataset_dict['path_rel_rgb'] if 'path_rel_rgb' in dataset_dict else [-1]
                }

                features = [rgb]
                if self.num_tasks > 1:
                    depth = self.post_process(dataset_dict["depth"]) if self.use_post_process else dataset_dict["depth"]
                    dict_output['depth'] = depth
                    features.append(depth)
                
                if self.num_tasks > 2:
                    normal = self.post_process(dataset_dict["normal"]) if self.use_post_process else dataset_dict["normal"]
                    dict_output['normal'] = normal
                    features.append(normal)
                
                if self.num_tasks > 3:
                    mask = self.post_process(dataset_dict["mask"]) if self.use_post_process else dataset_dict["mask"]
                    dict_output['mask'] = mask
                    features.append(mask)

                if self.use_repa_reg:
                    dict_output.update(self.get_dino_feature(meta))
                dict_output['x'] = torch.cat(features, dim=0)
                return dict_output
            except Exception as e:
                index += 1
                errors.append(f"{meta.get('path_rel_rgb', index)}: {e}")
        raise RuntimeError(
            "Unable to load any sample after a full dataset pass. "
            f"First errors: {errors[:3]}"
        )
