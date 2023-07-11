# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py  # noqa: E501
import functools
import os
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

#https://blog.csdn.net/qq_14898613/article/details/54577442
def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None: #返回用于启动进程start方法的名称
        mp.set_start_method('spawn')    #
    if launcher == 'pytorch':   #如果启动平台是pytorch
        _init_dist_pytorch(backend, **kwargs)   #初始化pytorch的启动
    elif launcher == 'slurm': #如果是启动平台是slurm
        _init_dist_slurm(backend, **kwargs) #初始化slurm的启动
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count() #获得可用gpu数量
    torch.cuda.set_device(rank % num_gpus)  #将模型加载到多张gpu上
    dist.init_process_group(backend=backend, **kwargs)  #初始化默认分布式进程组，这也将初始化分布包，backend是后端，根据构建的配置，如果是nccl则每个人进程对其使用的每个Gpu具有独占的访问权限


def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def get_dist_info():
    if dist.is_available():#如果分发的包可用，则返回
        initialized = dist.is_initialized() #检查默认进程组是否已初始化
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank() #返回在提供的组里的循环进程的rank
        world_size = dist.get_world_size() #返回进程的数量在循环过程组中
    else:
        rank = 0
        world_size = 1
    return rank, world_size #返回一个rank和进程的数量


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
