import torch

import data
import model
import loss
from option import args
from trainer_vd import Trainer_VD
from logger import logger
from thop import profile
import numpy as np
import torch.nn.functional as F


torch.manual_seed(args.seed)
chkp = logger.Logger(args)


#loader = data.Data(args)
model = model.Model(args, chkp).cuda()
from thop import profile
input = x = torch.ones(1, 3, 256, 256).cuda()
flops, params = profile(model, inputs=(input, ))
print(flops/1e9,params/1e6)
