import argparse
import random
import torch
import yaml
from collections import OrderedDict
from os import path as osp

from basicsr.utils import set_random_seed
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def _postprocess_yml_value(value):
    # None
    if value == '~' or value.lower() == 'none':
        return None
    # bool
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    # !!float number
    if value.startswith('!!float'):
        return float(value.replace('!!float', ''))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
        return float(value)
    # list
    if value.startswith('['):
        return eval(value)
    # str
    return value


def parse_options(root_path, is_train=True):
    """
    Python中的*与**操作符使用最多的就是两种用法。
    1.用做运算符，即*表示乘号，**表示次方。
    2.用于指定函数传入参数的类型的。*用于参数前面，表示传入的多个参数将按照元组的形式存储，是一个元组；
    **用于参数前则表示传入的(多个)参数将按照字典的形式存储，是一个字典。

    *args必须要在**kwargs,否则将会提示语法错误"SyntaxError: non-keyword arg after keyword arg."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()

    # parse yml to dict 解析yml到字典中
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])#解析文件流中的第一个YAML文件生成相应的python对象

    # distributed settings  分布式的设置
    if args.launcher == 'none': #launcher是none
        opt['dist'] = False #dist设置为False
        print('Disable distributed.', flush=True) #报错
    else:   #如果不是none
        opt['dist'] = True  #dist设置为True，之后对其他的进行设置
        if args.launcher == 'slurm' and 'dist_params' in opt:   #如果launcher是slurm并且dist_params在opt中
            init_dist(args.launcher, **opt['dist_params']) #将launcher，和opt按字典的形式传入到init_dist中然后进行数据分配到各个显卡上
        else:   #如果是pytorch
            init_dist(args.launcher) #将launcher，和opt按字典的形式传入到init_dist中然后进行数据分配到各个显卡上
    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed 设置seed
    seed = opt.get('manual_seed')   #从opt中取出manual_seed作为seed
    if seed is None:    #如果seed是不存在的
        seed = random.randint(1, 10000) #从1到10000中随机一个
        opt['manual_seed'] = seed   #将这个随机的结果赋值给manual
    set_random_seed(seed + opt['rank']) #

    # force to update yml options 强制更新yml选项
    if args.force_yml is not None:
        for entry in args.force_yml:
            # now do not support creating new keys
            keys, value = entry.split('=')
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = 'opt'
            for key in keys.split(':'):
                eval_str += f'["{key}"]'
            eval_str += '=value'
            # using exec function
            exec(eval_str)

    opt['auto_resume'] = args.auto_resume
    opt['is_train'] = is_train

    # debug setting debug设置
    if args.debug and not opt['name'].startswith('debug'):
        opt['name'] = 'debug_' + opt['name']

    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    # datasets
    for phase, dataset in opt['datasets'].items():
        # for multiple datasets, e.g., val_1, val_2; test_1, test_2
        phase = phase.split('_')[0] #输出_之前的内容，https://blog.csdn.net/qq_41780295/article/details/88555183·
        dataset['phase'] = phase #
        if 'scale' in opt:
            dataset['scale'] = opt['scale'] #将yml的scale赋值给scale
        if dataset.get('dataroot_gt') is not None:  #如果dataset中没有dataroot_gt
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt']) #将dataroot_gt的上级目录展开，提供绝对路劲
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key): #如果值不为空而且resume_state或者pretrain_network在key中
            opt['path'][key] = osp.expanduser(val) #展示val的绝对路径

    if is_train: #如果是训练模式
        experiments_root = osp.join(root_path, 'experiments', opt['name'])#将训练路径和experiments和name连接起来
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = osp.join(root_path, 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt, args #返回整理好的yml转字典信息和args的输入


@master_only
def copy_opt_file(opt_file, experiments_root):
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile
    cmd = ' '.join(sys.argv)
    filename = osp.join(experiments_root, osp.basename(opt_file))
    copyfile(opt_file, filename)

    with open(filename, 'r+') as f:
        lines = f.readlines()
        lines.insert(0, f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n')
        f.seek(0)
        f.writelines(lines)
