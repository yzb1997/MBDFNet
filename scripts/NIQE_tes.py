#coding=utf-8
from basicsr.metrics.niqe import calculate_niqe
import cv2
import os
from gauss import handle_dir
from tqdm import tqdm

def NIQE(model_name,
         save_path,
         deblur_path):
    # results of models in deblur datasets
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = os.path.join(save_path, f'{model_name}.txt')
    # utils.mkdir(file_name)
    save_performance_dir = open(file_name, 'w+')

    print(model_name)
    save_performance_dir.write(model_name)
    save_performance_dir.write('\n')

    print(deblur_path)
    file_list_sr = sorted(os.listdir(deblur_path))

    print(len(deblur_path))
    NIQE_v = 0

    # pbar_img_file = tqdm(total=len(file_list_sr), unit='image')
    for img_file in file_list_sr:
        result = cv2.imread(os.path.join(deblur_path,img_file))

        a = calculate_niqe(result, 0)

        save_performance_dir.write('{}.png NIQE: {}'.format(img_file, a))
        save_performance_dir.write('\n')

        print('{} NIQE: {}'.format(img_file, a))
        NIQE_v = NIQE_v + a

        # pbar_img_file.update(1)
        # pbar_img_file.set_description(f'Blur {img_file}')

    print(model_name, NIQE_v / len(file_list_sr))
    save_performance_dir.write(f'{model_name}, {NIQE_v / len(file_list_sr)}')


if __name__ == '__main__':
    NIQE_dir_path = '/data2/yangzhongbao/code/vivo_code/output/Train_on_GOPRO/Train_on_GOPRO_3'
    NIQE_datasets = os.listdir(NIQE_dir_path)
    NIQE_result_dir = '/data2/yangzhongbao/code/vivo_code/output/Train_on_GOPRO_results'
    handle_dir(NIQE_result_dir)

    ## for datasets squence
    # for dataset in NIQE_datasets:
    #     NIQE_dataset_path = os.path.join(NIQE_dir_path, dataset)
    #     NIQE(dataset, NIQE_result_dir, NIQE_dataset_path)
    
    
    ## for single dataset    
    NIQE('ICME', NIQE_result_dir, '/data2/yangzhongbao/code/vivo_code/output/ICME')             
             
             
             
                                                                                                                
