import torch
import os
from cv2 import imread


class ZurichRAW2RGB(torch.utils.data.Dataset):
    """ Canon RGB images from the "Zurich RAW to RGB mapping" dataset. You can download the full
    dataset (22 GB) from http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset. Alternatively, you can only download the
    Canon RGB images (5.5 GB) from https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip
    """
    def __init__(self, root):
        super().__init__()
        self.img_pth = root
        split = self.img_pth.split('/')[-2]

        self.image_list = self._get_image_list(split)

    def _get_image_list(self, split):
        if split == 'train':
            image_list = ['{:d}.jpg'.format(i) for i in range(46839)]
        elif split == 'test':
            image_list = ['{:d}.jpg'.format(i) for i in range(1204)]
        else:
            raise Exception

        return image_list

    def _get_image(self, im_id):
        path = os.path.join(self.img_pth, self.image_list[im_id])
        img = imread(path)
        return img

    def get_image(self, im_id):
        frame = self._get_image(im_id)

        return frame

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        frame = self._get_image(index)

        return frame
