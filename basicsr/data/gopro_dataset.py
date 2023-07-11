import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data
import torchvision.transforms.functional as TF

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.flow_util import dequantize_flow
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class GOPRODataset(data.Dataset):

    def __init__(self, opt):
        super(GOPRODataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])  # 转换成路径形式
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.data_info = {'lq_path': [], 'gt_path': []}

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:  # 打开meta_info_file文件路径并且处理异常
            for line in fin:
                folder, frame_num, _ = line.split(' ')  # 对字符进行分割取出文件夹名称，帧的编号
                self.keys.extend([f'{folder}/{i:06d}' for i in range(int(frame_num))])  # 在Keys列表中增加
        # print(self.keys)
        # file client (io backend) 文件客户端
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        # print(index)
        scale = self.opt['scale']
        key = self.keys[index]

        clip_name, frame_name = key.split('/')  # key example: 000/00000000
        # print(key)
        # print(clip_name, frame_name)
        # get the GT frame (as the center frame)    获得对照帧作为中心帧
        if self.is_lmdb:
            img_lq_path = img_gt_path = f'{clip_name}/{frame_name}'
        else:
            img_gt_path = self.gt_root / clip_name  /f'{frame_name}.png'
            img_lq_path = self.lq_root / clip_name / f'{frame_name}.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        img_bytes = self.file_client.get(img_lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training 训练增强
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, img_gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets在验证或测试中裁剪不匹配的GT图像，特别是对SR基准数据集
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            # img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
            if self.opt['patch_size'] is not None:
                patch_size = self.opt['patch_size']
                img_lq = TF.center_crop(img_lq, (patch_size,patch_size))
                img_gt = TF.center_crop(img_gt, (patch_size,patch_size))


        # normalize
        # if self.mean is not None or self.std is not None:
        #     normalize(img_lq, self.mean, self.std, inplace=True)
        #     normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)

@DATASET_REGISTRY.register()
class GOPRORecurrentDataset(data.Dataset):

    def __init__(self, opt):
        super(GOPRORecurrentDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']

        self.keys = []
        if opt['test_mode']:
            with open(opt['meta_info_file_test'], 'r') as fin:
                for line in fin:
                    folder, frame_num, _ = line.split(' ')
                    self.keys.extend([f'{folder}/{i:06d}/{frame_num}' for i in range(int(frame_num))])
        else:
            with open(opt['meta_info_file_train'], 'r') as fin:
                for line in fin:
                    folder, frame_num, _ = line.split(' ')
                    self.keys.extend([f'{folder}/{i:06d}/{frame_num}' for i in range(int(frame_num))])

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]

        clip_name, frame_name, frame_num = key.split('/')  # key example: 000/000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        if start_frame_idx > int(frame_num) - self.num_frame:
            start_frame_idx = random.randint(0, int(frame_num) - self.num_frame)
        end_frame_idx = start_frame_idx + self.num_frame

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:06d}'
                img_gt_path = f'{clip_name}/{neighbor:06d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:06d}.png'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:06d}.png'

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)
