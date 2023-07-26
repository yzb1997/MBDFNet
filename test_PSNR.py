from basicsr.metrics.psnr_ssim import calculate_psnr,calculate_ssim
import torch
import numpy as np
import math
import cv2
import os
import utils


def PSNR(model_name,
         path):
    save_path = '/data2/yangzhongbao/code/ICME_experiments/PSNR_results'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = os.path.join(save_path, f'{model_name}.txt')
    # utils.mkdir(file_name)
    save_performance_dir = open(file_name, 'w+')

    print(model_name)
    save_performance_dir.write(model_name)
    save_performance_dir.write('\n')
    SRlist=[]
    GTlist=[]

    path_gt = '/data2/yangzhongbao/datasets/deblur/REDS/REDSX2X4X10_CRF_IMG/test/gt/'

    print(path)
    file_list_sr = os.listdir(path)
    file_list_gt = os.listdir(path_gt)

    print(len(file_list_gt))
    PSNR_v =0
    SSIM_v = 0
    frame_num = 0

    if len(file_list_sr)!=11:
        for img_file in sorted(file_list_sr):
            result = cv2.imread(path+img_file)
            gt = cv2.imread(path_gt + img_file)

            a = calculate_psnr(result,gt,0)
            b = calculate_ssim(result,gt,0)

            save_performance_dir.write('{}.png PSNR: {}'.format(img_file, a))
            save_performance_dir.write('\n')

            print('{}.png PSNR: {}'.format(img_file, a))
            PSNR_v = PSNR_v+a
            SSIM_v = SSIM_v+b

        print(model_name, PSNR_v/len(file_list_gt), SSIM_v/len(file_list_gt))
        save_performance_dir.write(f'{model_name}, {PSNR_v/len(file_list_gt)}, {SSIM_v/len(file_list_gt)}')

    else:
        for frames in file_list_sr:
            results_path = os.path.join(path, frames)
            gt_path = os.path.join(path_gt, frames)
            frame_path = sorted(os.listdir(results_path))

            for img_file in frame_path:
                result_img_path = os.path.join(results_path, img_file)
                gt_img_path = os.path.join(gt_path, img_file)

                result = cv2.imread(result_img_path)
                gt = cv2.imread(gt_img_path)

                a = calculate_psnr(result,gt,0)
                b = calculate_ssim(result,gt,0)

                save_performance_dir.write('{} PSNR: {}'.format(img_file, a))
                save_performance_dir.write('\n')

                print('{}.png PSNR: {}'.format(img_file, a))
                PSNR_v = PSNR_v + a
                SSIM_v = SSIM_v + b

            frame_num = frame_num + len(frame_path)



        print(model_name, PSNR_v/frame_num, SSIM_v/frame_num)
        save_performance_dir.write(f'{model_name}, {PSNR_v/len(file_list_gt)}, {SSIM_v/len(file_list_gt)}')

if __name__ == '__main__':
     PSNR('MBDFNet_test',
          '/data2/yangzhongbao/code/vivo_code/test/baseline_same_stage/')





