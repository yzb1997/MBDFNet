from torch.utils import data as data
from torchvision.transforms.functional import normalize
from pathlib import Path

import torchvision.transforms.functional as TF
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.zurich_raw2rgb_dataset import ZurichRAW2RGB
from basicsr.utils.synthetic_burst_generation import rgb2rawburst, random_crop
import torchvision.transforms as tfm
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class BrustImageDataset(data.Dataset):
    """ Generates a synthetic burst
    args:
        index: Index of the image in the base_dataset used to generate the burst

    returns:
        burst: Generated LR RAW burst, a torch tensor of shape
               [burst_size, 4, self.crop_sz / (2*self.downsample_factor), self.crop_sz / (2*self.downsample_factor)]
               The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer mosaick.
               The extra factor 2 in the denominator (2*self.downsample_factor) corresponds to the mosaicking
               operation.

        frame_gt: The HR RGB ground truth in the linear sensor space, a torch tensor of shape
                  [3, self.crop_sz, self.crop_sz]

        flow_vectors: The ground truth flow vectors between a burst image and the base image (i.e. the first image in the burst).
                      The flow_vectors can be used to warp the burst images to the base frame, using the 'warp'
                      function in utils.warp package.
                      flow_vectors is torch tensor of shape
                      [burst_size, 2, self.crop_sz / self.downsample_factor, self.crop_sz / self.downsample_factor].
                      Note that the flow_vectors are in the LR RGB space, before mosaicking. Hence it has twice
                      the number of rows and columns, compared to the output burst.

                      NOTE: The flow_vectors are only available during training for the purpose of using any
                            auxiliary losses if needed. The flow_vectors will NOT be provided for the bursts in the
                            test set

        meta_info: A dictionary containing the parameters used to generate the synthetic burst."""

    def __init__(self, opt):
        super(BrustImageDataset, self).__init__()
        self.opt = opt
        self.io_backend_opt = opt['io_backend']
        self.gt_root = opt['dataroot_gt'] #转换成路径形式
        self.io_backend_opt = opt['io_backend']
        # self.gt_root = '../datasets/synthetic/test/canon'
        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.gt_folder, line.split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))


        self.base_dataset = ZurichRAW2RGB(root=self.gt_root)

        self.burst_size = 14
        self.transform = tfm.ToTensor()

        self.downsample_factor = 4
        self.burst_transformation_params = {'max_translation': 24.0,
                                            'max_rotation': 1.0,
                                            'max_shear': 0.0,
                                            'max_scale': 0.0,
                                            'border_crop': 24}

        self.image_processing_params = {'random_ccm': True, 'random_gains': True, 'smoothstep': True,
                                        'gamma': True,
                                        'add_noise': True}
        self.interpolation_type = 'bilinear'



    def __getitem__(self, index):
        self.flag = self.gt_root.split('/')[-2]

        if self.flag == 'train':
            self.crop_sz = 256
        elif self.flag == 'test':
            self.crop_sz = 384

        frame = self.base_dataset[index]
        gt_path = self.paths[index]
        # Augmentation, e.g. convert to tensor
        if self.transform is not None:
            # frame = Image.fromarray(frame)
            frame = self.transform(frame)

        # Extract a random crop from the image
        if self.flag == 'train':
            self.crop_sz = 256
            crop_sz = self.crop_sz + 2 * self.burst_transformation_params.get('border_crop', 0)
            frame_crop = random_crop(frame, crop_sz)
        elif self.flag == 'test':
            self.crop_sz = 384
            crop_sz = self.crop_sz + 2 * self.burst_transformation_params.get('border_crop', 0)
            frame_crop = random_crop(frame, crop_sz)
            frame_crop = TF.center_crop(frame, (crop_sz,crop_sz))


        # Generate RAW burst
        burst, frame_gt, burst_rgb, flow_vectors, meta_info = rgb2rawburst(frame_crop,
                                                                           self.burst_size,
                                                                           self.downsample_factor,
                                                                           burst_transformation_params=self.burst_transformation_params,
                                                                           image_processing_params=self.image_processing_params,
                                                                           interpolation_type=self.interpolation_type
                                                                           )

        if self.burst_transformation_params.get('border_crop') is not None:
            border_crop = self.burst_transformation_params.get('border_crop')
            frame_gt = frame_gt[:, border_crop:-border_crop, border_crop:-border_crop]


        return {'burst': burst, 'gt_path': gt_path,'frame_gt': frame_gt, 'flow_vectors': flow_vectors, 'meta_info': meta_info}
    
    def __len__(self):
        return len(self.base_dataset)

if __name__ == '__main__':
    BrustImageDataset = BrustImageDataset()
    print(BrustImageDataset)
