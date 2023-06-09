import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange, repeat

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
import random
from .base_model import BaseModel

@MODEL_REGISTRY.register()
class mimo_pan_Model_075_s1(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):    #初始化SRModel类，比如定义网络和load weight
        super(mimo_pan_Model_075_s1, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()   #

    def init_training_settings(self):  #初始化与训练相关的，比如loss， 设置optmizers和schedulers
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('l1_opt'):
            self.l1_pix = build_loss(train_opt['l1_opt']).to(self.device)
        else:
            self.l1_pix = None

        if train_opt.get('fq_opt'):
            self.fq_pix = build_loss(train_opt['fq_opt']).to(self.device)
        else:
            self.fq_pix = None
        if train_opt.get('pfq_opt'):
            self.pfq_pix = build_loss(train_opt['pfq_opt']).to(self.device)
        else:
            self.pfq_pix = None
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('ct_opt'):
            self.ct_loss = build_loss(train_opt['ct_opt']).to(self.device)
        else:
            self.ct_loss = None

        if train_opt.get('HEM_opt'):
            self.hem_loss = build_loss(train_opt['HEM_opt']).to(self.device)
        else:
            self.hem_loss = None

        if train_opt.get('BD_opt'):
            self.bd_loss = build_loss(train_opt['BD_opt']).to(self.device)
        else:
            self.bd_loss = None

        if train_opt.get('TVloss_opt'):
            self.tv_loss = build_loss(train_opt['TVloss_opt']).to(self.device)
        else:
            self.tv_loss = None

        if train_opt.get('Gradientloss_opt'):
            self.gr_loss = build_loss(train_opt['Gradientloss_opt']).to(self.device)
        else:
            self.gr_loss = None

        if train_opt.get('ssim_opt'):
            self.ssim_loss = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.ssim_loss = None

##################################################################################################
# '''
#     losses of Deep-RFT
#     CharbonnierLoss
#     EdgeLoss
#     fftLoss
# '''
        if train_opt.get('msc_opt'):
            self.msc_pix = build_loss(train_opt['msc_opt']).to(self.device)
        else:
            self.msc_pix = None

        if train_opt.get('msed_opt'):
            self.msed_pix = build_loss(train_opt['msed_opt']).to(self.device)
        else:
            self.msed_pix = None

        if train_opt.get('msfr_opt'):
            self.msfr_pix = build_loss(train_opt['msfr_opt']).to(self.device)
        else:
            self.msfr_pix = None

        # if self.cri_pix is None and self.cri_perceptual is None:
        #     raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self): #具体设置optimizer，可以根据实际需求，对params设置多组不同的optimizer
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):  #提供数据，是与dataloadre（dataset）的接口
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):    #优化参数，即一个完整的train的step，包括了forward，loss的计算，backward，参数优化等
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        # use REDS video dataset
        # self.gt1 = rearrange(self.gt, 'b t c h w -> (b t) c h w')
        # self.gt2 = F.interpolate(self.gt1, scale_factor=0.5, mode='bilinear')
        # self.gt3 = F.interpolate(self.gt1, scale_factor=0.25, mode='bilinear')
        #
        # self.output1 = rearrange(self.output[0], 'b t c h w -> (b t) c h w')
        # self.output2 = rearrange(self.output[1], 'b t c h w -> (b t) c h w')
        # self.output3 = rearrange(self.output[2], 'b t c h w -> (b t) c h w')


        self.input1 = self.lq

        self.gt1 = self.gt

        self.output1 = self.output


        #############################################################
        #loss stages of MIMO-UNet
        l_total = 0
        l_total_l1 = 0
        l_total_fq = 0
        l_total_ct = 0
        l_total_hem = 0
        l_total_tv = 0
        l_total_gr = 0
        l_total_bd = 0
        l_total_ssim = 0
        loss_dict = OrderedDict()
        if self.l1_pix:
            l_l1_1 = self.l1_pix(self.output1, self.gt1)
            l_total_l1 = l_l1_1
            loss_dict['l_l1_1'] = l_l1_1


        if self.fq_pix:
            l_fq_1 = self.fq_pix(self.output1, self.gt1)
            l_total_fq = l_fq_1
            loss_dict['l_fq_1'] = l_fq_1



        l_total = l_total + l_total_l1 + l_total_fq
        loss_dict['l_total'] = l_total

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self): #测试流程
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):   #validation的流程，分了dist（多卡一起做validation）和nondist（只有单卡做validation）
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):    ##
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', True)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            if 'key' in val_data.keys():
                img_name = osp.splitext(osp.basename(val_data['key'][0]))[0]
            else:
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], str(dataset_name),
                                                 f'{str(img_name)}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], str(dataset_name),
                                                 f'{str(img_name)}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger): #控制如可打印validation的结果
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):  #得到网络的输出的结果，这个函数会在validation中用到（实际可以简化）
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()

        # 判断网络输出是否是单一输出，多尺度输出为的话网络输出结果为tuple，只取第一个，其他用作loss

        # if type(self.output) == tuple:
            # self.input1 = self.lq
            # self.input2 = F.interpolate(self.input1, scale_factor=0.8, mode='bicubic')
            # self.input3 = F.interpolate(self.input2, scale_factor=0.5, mode='bicubic')


            # out_dict['result'] = self.output[0].detach().cpu()
            # out_dict['result'] = self.input3.detach().cpu()
        # else:
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):    #保存网络（.pth）以及训练状态
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

