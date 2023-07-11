import time
import torch
import basicsr.archs.vivo_stage3_final_v49_bdmlp_arch as mynet

input=torch.randn(1, 3, 256, 256).cuda()
timeCount=0
for i in range(100):
    start_time=time.time()
    forward = mynet.DEBLUR_stage3_final_V49_bdmlp(0.75).cuda()
    end_time= time.time()
    running_time = end_time-start_time
    timeCount+=running_time

print(f'running time: {timeCount/100}')
