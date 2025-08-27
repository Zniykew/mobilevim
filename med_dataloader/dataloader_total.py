
from PIL import Image 
import pandas as pd
import med_dataloader.transforms as transforms
import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import matplotlib.image as mping

to_tensor = transforms.ToTensor()

def load_rgb(path, dataset_type):
    if dataset_type == 'blender':
        rgb = Image.open(path)
        return rgb
    else:
        rgb = imread(path).astype(np.float32)
        rgb = rgb[:, :, :3]
        return rgb

def load_depth(path, dataset_type):
    if dataset_type == 'smalldata':
        depth = imread(path).astype(np.float32)
        return depth[:, :, 0]  # smalldata深度只取第一通道
    elif dataset_type == 'colon':
        depth = mping.imread(path).astype(np.float32)  # 自动归一化
        return depth * 20.0  # 放缩到0~20
    elif dataset_type == 'blender':
        depth = np.load(path)
        return depth

class MyDataloader(data.Dataset):
    def __init__(self, root, type, dataset):
        self.root = Path(root)
        self.type = type
        self.dataset = dataset
        self.transform = self.train_transform if self.type == 'train' else self.val_transform

        # 读取所有图片地址
        if dataset == 'colon':
            self._init_colon()
        elif dataset == 'smalldata':
            self._init_smalldata()
        elif dataset == 'blender':
            self._init_blender()
        else:
            raise RuntimeError(f"Invalid dataset type: {dataset}\nSupported datasets are: colon, smalldata, blender")
        
    def _init_colon(self):
        if self.type == 'train':
            scene_list_path = self.root / 'train_T1_L1.txt'
            self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]  # [:-1]吃回车，txt文件每行以回车结尾（注意最后一行也是!）
        elif self.type == 'val':
            scene_list_path = self.root / 'val_T1_L1.txt'
            self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]
        self.crawl_folders()  # 获取目标文件夹中每张图片地址

    def _init_smalldata(self):
        if self.type == 'train':
            scene_list_path = self.root / 'train.txt'
            self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]
        elif self.type == 'val':
            scene_list_path = self.root / 'val.txt'
            self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]
        self.crawl_folders()
        
    def _init_blender(self):   
        if self.type == 'train':  
            csv_path = self.root / 'train.csv'  
        elif self.type == 'val':  
            csv_path = self.root / 'val.csv'

        self.data = pd.read_csv(csv_path)
        self.imgs = [{'RGB': str(self.root / row['image']), 'GT': str(self.root / row['depth'])} for _, row in self.data.iterrows()]  # 地址集合
        # print(self.imgs[0])
  
    def crawl_folders(self):  # for colon,smalldata
        sequence_set = []
        for scene_path in self.scenes: # 遍历self.scenes中的每个场景
            if not isinstance(scene_path, Path):  
                scene_path = Path(scene_path)  
            
            if self.dataset == 'colon':
                img_files = sorted(scene_path.files('FrameBuffer_*.png'))  # rgb图片地址清单
                depth_files = sorted(scene_path.files('Depth_*.png'))  # depth图片地址清单
            elif self.dataset == 'smalldata':
                img_files = sorted(scene_path.files('*.png'))
                depth_path = scene_path / 'GT'
                depth_files = sorted(depth_path.files('*.png'))
    
            if len(img_files) != len(depth_files):  
                print(f"Warning: Mismatched number of images and depth maps in {scene_path}")  
                assert False  
    
            for img_file, depth_file in zip(img_files, depth_files):   
                sample = {'RGB': str(img_file), 'GT': str(depth_file)}  
                sequence_set.append(sample)  
  
        self.imgs = sequence_set  # 地址集合
        # print(self.imgs[0])

    def __getitem__(self, index):
        
        sample = self.imgs[index]
        rgb = load_rgb(sample['RGB'], self.dataset)
        depth = load_depth(sample['GT'], self.dataset)

        if self.transform is not None:
            rgb_tensor, depth_tensor = self.transform(rgb, depth)
        else:
            raise(RuntimeError("transform not defined"))

        depth_tensor = depth_tensor.unsqueeze(0)  # The shape of GT in colon dataset is [256, 256]

        return {'image':rgb_tensor, 'depth': depth_tensor, 'has_valid_depth': True}

    def __len__(self):
        return len(self.imgs)
    
    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(self, rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))
    

class MedicalDataloader(MyDataloader):
    def __init__(self, root, type, dataset):
        super(MedicalDataloader, self).__init__(root, type, dataset)

    def train_transform(self, rgb, depth):
        rgb_tensor = to_tensor(np.asfarray(rgb, dtype='float') / 255)
        depth_tensor = to_tensor(depth)
        return rgb_tensor, depth_tensor

    def val_transform(self, rgb, depth):
        rgb_tensor = to_tensor(np.asfarray(rgb, dtype='float') / 255)
        depth_tensor = to_tensor(depth)
        return rgb_tensor, depth_tensor