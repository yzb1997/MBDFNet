# ddp train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt /data2/yangzhongbao/code/vivo_code/BasicSR/options/train/MultiScaleNet/20221127_ICME_VIVO_V49_stage4Kernel5To3ReLUToGeLU_Gmlp_BS52_PS256_LR3E_4_GoPro_COS_ITER60_L1_FFT_ddp.yml --launcher pytorch

# ddp debug
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt /data2/yangzhongbao/code/vivo_code/BasicSR/options/train/MultiScaleNet/20221130_ICME_VIVO_V49_stage4Kernel5To3ReLUToGeLU_Gmlp_BS52_PS256_LR3E_4_mydatasets_COS_ITER60_L1_FFT_ddp.yml --launcher pytorch --debug

# train
# CUDA_VISIBLE_DEVICES=4 python train.py -opt /data2/yangzhongbao/code/vivo_code/BasicSR/options/train/MultiScaleNet/20221130_ICME_VIVO_V49_stage4Kernel5To3ReLUToGeLU_oneOutput_BS8_PS256_LR3E_4_GoPro_COS_ITER60_L1_FFT.yml

# debug
# CUDA_VISIBLE_DEVICES=4 python train.py -opt /data2/yangzhongbao/code/vivo_code/BasicSR/options/train/MultiScaleNet/20221130_ICME_VIVO_V49_stage4Kernel5To3ReLUToGeLU_oneOutput_BS8_PS256_LR3E_4_GoPro_COS_ITER60_L1_FFT.yml --debug
