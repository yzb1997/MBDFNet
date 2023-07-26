import os
import torch
import argparse
from torch.backends import cudnn
from models.MBDFNet.MBDFNet_arch import MBDFNet as mynet
from train import _train
from eval import _eval

def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = mynet(0.75)
    # print(model)
    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='baseline_same_stage', type=str)
    parser.add_argument('--data_dir', type=str, default='/data2/yangzhongbao/datasets/deblur/REDS/REDSX2X4X10_CRF_IMG')
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)

    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=3000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--valid_freq', type=int, default=100)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 500 for x in range(3000//500)])

    # Test
    parser.add_argument('--test_model', type=str, default='/data2/yangzhongbao/code/MBDFNet/experiments/model.pth')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])

    args = parser.parse_args()
#    args.model_save_dir = os.path.join('results/', args.model_name, 'weights/')
    args.result_dir = os.path.join('/data2/yangzhongbao/code/vivo_code/test/', args.model_name)
    print(args)
    main(args)
