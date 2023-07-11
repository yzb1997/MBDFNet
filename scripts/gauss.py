import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import math
#
#
# def batch_calc_video_PSNR_SSIM(root_list, crop_border=4, shift_window_size=0, test_ycbcr=False, crop_GT=False,
#                                save_log=False, save_log_root=None, combine_save=False):
#     '''
#     required params:
#         root_list: a list, each item should be a dictionary that given two key-values:
#             output: the dir of output videos
#             gt: the dir of gt videos
#     optional params:
#         crop_border: defalut=4, crop pixels when calculating PSNR/SSIM
#         shift_window_size: defalut=0, if >0, shifting image within a window for best metric
#         test_ycbcr: default=False, if True, applying Ycbcr color space
#         crop_GT: default=False, if True, cropping GT to output size
#         save_log: default=False, if True, saving csv log
#         save_log_root: thr dir of output log
#         combine_save: default=False, if True, combining all output log to one csv file
#     return:
#         log_list: a list, each item is a dictionary that given two key-values:
#             data_path: the evaluated dir
#             log: the log of this dir
#     '''
#     if save_log:
#         assert save_log_root is not None, "Unknown save_log_root!"
#
#     total_csv_log = []
#     log_list = []
#     for i, root in enumerate(root_list):
#         ouput_root = root['output']
#         gt_root = root['gt']
#         print(">>>>  Now Evaluation >>>>")
#         print(">>>>  OUTPUT: {}".format(ouput_root))
#         print(">>>>  GT: {}".format(gt_root))
#         csv_log, logs = calc_video_PSNR_SSIM(
#             ouput_root, gt_root, crop_border=crop_border, shift_window_size=shift_window_size,
#             test_ycbcr=test_ycbcr, crop_GT=crop_GT
#         )
#         log_list.append({
#             'data_path': ouput_root,
#             'log': logs
#         })
#
#         # output the PSNR/SSIM log of each evaluated dir to a single csv file
#         if save_log:
#             csv_log['row_names'] = [os.path.basename(p) for p in csv_log['row_names']]
#             write_csv(file_path=os.path.join(save_log_root, "{}_{}.csv".format(i, csv_log['row_names'][0])),
#                       data=np.array(csv_log['psnr_ssim']),
#                       row_names=csv_log['row_names'],
#                       col_names=csv_log['col_names'])
#             total_csv_log.append(csv_log)
#
#     # output all PSNR/SSIM log to a csv file
#     if save_log and combine_save and len(total_csv_log) > 0:
#         com_csv_log = {
#             'col_names': total_csv_log[0]['col_names'],
#             'row_names': [],
#             'psnr_ssim': []
#         }
#         for csv_log in total_csv_log:
#             com_csv_log['row_names'].append(csv_log['row_names'][0])
#             com_csv_log['psnr_ssim'].append(csv_log['psnr_ssim'][0])
#         write_csv(file_path=os.path.join(save_log_root, "psnr_ssim.csv"),
#                   data=np.array(com_csv_log['psnr_ssim']),
#                   row_names=com_csv_log['row_names'],
#                   col_names=com_csv_log['col_names'])
#
#     print("--------------------------------------------------------------------------------------")
#     for i, logs in enumerate(log_list):
#         print("## The {}-th:".format(i))
#         print(">> ", logs['data_path'])
#         for log in logs['log']:
#             print(">> ", log)
#
#     return log_list

def handle_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print("mkdir:", dir)


def matlab_style_gauss2D(shape=(5, 5), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def get_blur_kernel():
    gaussian_sigma = 1.0
    gaussian_blur_kernel_size = int(math.ceil(gaussian_sigma * 3) * 2 + 1)
    kernel = matlab_style_gauss2D((gaussian_blur_kernel_size, gaussian_blur_kernel_size), gaussian_sigma)
    return kernel


def get_blur(img, kernel):
    img = np.array(img).astype('float32')
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()

    kernel_size = kernel.shape[0]
    psize = kernel_size // 2
    img_tensor = F.pad(img_tensor, (psize, psize, psize, psize), mode='replicate')

    gaussian_blur = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, stride=1,
                              padding=int((kernel_size - 1) // 2), bias=False)
    nn.init.constant_(gaussian_blur.weight.data, 0.0)
    gaussian_blur.weight.data[0, 0, :, :] = torch.FloatTensor(kernel)
    gaussian_blur.weight.data[1, 1, :, :] = torch.FloatTensor(kernel)
    gaussian_blur.weight.data[2, 2, :, :] = torch.FloatTensor(kernel)

    blur_tensor = gaussian_blur(img_tensor)
    blur_tensor = blur_tensor[:, :, psize:-psize, psize:-psize]

    blur_img = blur_tensor[0].detach().numpy().transpose(1, 2, 0).astype('float32')

    return blur_img


def image_post_process_results(ori_root, save_root, alpha=3.):
    handle_dir(save_root)

    guassian_kernel = get_blur_kernel()

    frame_names = sorted(os.listdir(os.path.join(ori_root)))
    for fn in frame_names:
        ori_img = cv2.imread(os.path.join(ori_root, fn)).astype('float32')
        blur_img = get_blur(ori_img, guassian_kernel).astype('float32')

        res_img = ori_img - blur_img

        result = blur_img + alpha * res_img

        basename = fn.split(".")[0]
        cv2.imwrite(os.path.join(save_root, "{}_post.png".format(basename)), result)

        ##
        # res_img = np.clip(res_img, 0, np.max(res_img))
        # res_img = (res_img / np.max(res_img)) * 255.
        # cv2.imwrite(os.path.join(save_root, "{}_res.png".format(basename)), res_img)
        ##

        print("{} done!".format(fn))


def video_post_process_results(ori_root, save_root, alpha=3.):
    handle_dir(save_root)

    guassian_kernel = get_blur_kernel()

    video_names = os.listdir(ori_root)
    video_names.sort(key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2].split('.')[0])))

    for vn in video_names:
        frame_names = os.path.join(ori_root, vn)

        #
        ori_img = cv2.imread(frame_names).astype('float32')
        blur_img = get_blur(ori_img, guassian_kernel).astype('float32')
        #
        res_img = ori_img - blur_img
        #
        result = blur_img + alpha * res_img
        #
        basename = vn.split(".")[0]
        cv2.imwrite(os.path.join(save_root, "{}_post.png".format(basename)), result)
        #
        # ##
        # # res_img = np.clip(res_img, 0, np.max(res_img))
        # # res_img = (res_img / np.max(res_img)) * 255.
        # # cv2.imwrite(os.path.join(save_root, vn, "{}_res.png".format(basename)), res_img)
        # ##
        #
        print("{} done!".format(vn))


if __name__ == '__main__':
    # image_post_process_results(
    #     ori_root='/media/csbhr/Bear/Dataset/FaceSR/face/test/bic',
    #     save_root='./temp/edge/residual',
    #     alpha=2
    # )

    root_list = []
    for i in range(21):
        alpha = 4
        save_root = r'D:\windows\yzb\project\Python\pytorch\BasicSR-master\datasets\VIVO\1111\post_{}'.format(alpha)
        video_post_process_results(
            ori_root= r'D:\windows\yzb\project\Python\pytorch\BasicSR-master\results\vivo_kls_model_0521_haarscff_allHIDE_fftloss_31.5w\visualization\vivo_kls_model_0521_haarscff_allHIDE_fftloss_31.5w',
            save_root=save_root,
            alpha=alpha
        )
        # root_list.append({
        #     'output': save_root,
        #     'gt': '/home/csbhr/Disk-2T/work/OpenUtility/temp/edge_sharpen/HR'
        # })
    # batch_calc_video_PSNR_SSIM(root_list)