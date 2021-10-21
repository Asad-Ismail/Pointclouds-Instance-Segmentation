
import torch
import torch.optim as optim
import time, sys, os, random
from tensorboardX import SummaryWriter
import numpy as np
import glob
from util.config import cfg
from util.log import logger
import util.utils as utils
import torch.distributed as dist
from data_loader_util import synchronize, get_rank
from checkpoint import strip_prefix_if_present, align_and_update_state_dicts
from checkpoint import checkpoint
from solver import PolyLR
from torch.optim.lr_scheduler import StepLR

def init():
    # copy important files to backup
    backup_dir = os.path.join(cfg.exp_path, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp train.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    # log the config
    logger.info(cfg)

    # summary writer
    global writer
    writer = SummaryWriter(cfg.exp_path)
    gpu=0
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.enabled=False
    # random seed
    # random.seed(cfg.manual_seed)
    # np.random.seed(cfg.manual_seed)
    # torch.manual_seed(cfg.manual_seed)
    # torch.cuda.manual_seed_all(cfg.manual_seed)

def train(train_loader, model, model_fn, optimizer, start_iter, scheduler, save_to_disc=True):
    iter_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    am_dict = {}

    model.train()
    start_time = time.time()
    end = time.time()
    data_len = len(train_loader)
    for iteration, batch in enumerate(train_loader, start_iter):
        data_time.update(time.time() - end)
        torch.cuda.empty_cache()
        # epoch = int(iteration / data_len)
        ##### adjust learning rate
        # utils.step_learning_rate(optimizer, cfg.lr, epoch - 1, cfg.step_epoch, cfg.multiplier)

        ##### prepare input and forward
        loss, _, visual_dict, meter_dict = model_fn(batch, model, iteration)

        ##### meter_dict
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = utils.AverageMeter()
            am_dict[k].update(v[0], v[1])

        ##### backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        ##### time and print
        # current_iter = (epoch - 1) * len(train_loader) + i + 1
        current_iter = iteration
        max_iter = len(train_loader)
        remain_iter = max_iter - current_iter

        iter_time.update(time.time() - end)
        end = time.time()

        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        if save_to_disc:
            if iteration > cfg.prepare_epochs:
                sys.stdout.write(
                    "iter: {}/{} lr: {:.6f} loss: {:.4f}({:.4f}) score_loss: {:.4f}({:.4f}) data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n".format
                    (iteration + 1, len(train_loader), scheduler.get_lr()[0], am_dict['loss'].val, am_dict['loss'].avg, am_dict['score_loss'].val, am_dict['score_loss'].avg,
                     data_time.val, data_time.avg, iter_time.val, iter_time.avg, remain_time=remain_time))
            else:
                sys.stdout.write(
                    "iter: {}/{} lr: {:.6f} loss: {:.4f}({:.4f}) data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n".format
                    (iteration + 1, len(train_loader), scheduler.get_lr()[0], am_dict['loss'].val, am_dict['loss'].avg,
                     data_time.val, data_time.avg, iter_time.val, iter_time.avg, remain_time=remain_time))

        if save_to_disc:
            logger.info("iteration: {}/{}, lr:{:.6f}  train loss: {:.4f}, time: {}s".format(iteration, len(train_loader), scheduler.get_lr()[0], am_dict['loss'].avg, time.time() - start_time))
            # if epoch % cfg.save_freq == 0:
                # utils.checkpoint_save(model, cfg.output_path, cfg.config.split('/')[-1][:-5], epoch, cfg.save_freq, use_cuda)
            if (iteration % cfg.save_freq == 0 or iteration==cfg.max_iter-1) and save_to_disc:
                checkpoint(model, optimizer, iteration, cfg.output_path, None, None)

            for k in am_dict.keys():
                if k in visual_dict.keys():
                    writer.add_scalar(k+'_train', am_dict[k].avg, iteration)


def eval_epoch(val_loader, model, model_fn, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    am_dict = {}

    with torch.no_grad():
        model.eval()
        start_epoch = time.time()
        for i, batch in enumerate(val_loader):

            ##### prepare input and forward
            loss, preds, visual_dict, meter_dict = model_fn(batch, model, epoch)

            ##### meter_dict
            for k, v in meter_dict.items():
                if k not in am_dict.keys():
                    am_dict[k] = utils.AverageMeter()
                am_dict[k].update(v[0], v[1])

            ##### print
            sys.stdout.write("\riter: {}/{} loss: {:.4f}({:.4f})".format(i + 1, len(val_loader), am_dict['loss'].val, am_dict['loss'].avg))
            if (i == len(val_loader) - 1): print()

        logger.info("epoch: {}/{}, val loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg, time.time() - start_epoch))

        for k in am_dict.keys():
            if k in visual_dict.keys():
                writer.add_scalar(k + '_eval', am_dict[k].avg, epoch)

if __name__ == '__main__':
    ##### init
    init()

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]


    ###
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()



    ##### model
    logger.info('=> creating model ...')

    #if model_name == 'pointgroup':
    from model.pointgroup.pointgroup import PointGroup as Network
    from model.pointgroup.pointgroup import model_fn_decorator
    #else:
    #    print("Error: no model - " + model_name)
    #    exit(0)

    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    cfg.use_syncbn=False

    if cfg.use_syncbn:
        print('use sync BN')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # logger.info(model)
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### optimizer
    if cfg.optim == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optim == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    scheduler = PolyLR(optimizer, max_iter=cfg.max_iter, power=0.9, last_step=-1)

    ###
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.local_rank], output_device=cfg.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False, find_unused_parameters=True
        )


    ##### model_fn (criterion)
    model_fn = model_fn_decorator()



    import os.path as osp
    start_iter = -1
    if cfg.pretrain:
        logger.info("=> loading checkpoint '{}'".format(cfg.pretrain))
        loaded_state_dict = torch.load(cfg.pretrain, map_location=torch.device("cpu"))['state_dict']
        model_state_dict = model.state_dict()
        #loaded_state_dict = strip_prefix_if_present(loaded, prefix="module.")
        #align_and_update_state_dicts(model_state_dict, loaded_state_dict)
        # Changed class size so load those which matches
        new_state_dict={k:v if v.size()==model_state_dict[k].size()  else  model_state_dict[k] for k,v in zip(model_state_dict.keys(), loaded_state_dict.values()) }
        model.load_state_dict(new_state_dict)
        logger.info("=> done loading")

    if cfg.resume:
        checkpoint_fn = cfg.resume
        if osp.isfile(checkpoint_fn):
            logger.info("=> loading checkpoint '{}'".format(checkpoint_fn))
            state = torch.load(checkpoint_fn, map_location=torch.device("cpu"))
            curr_iter = state['iteration'] + 1
            start_iter = curr_iter
            model_state_dict = model.state_dict()
            loaded_state_dict = strip_prefix_if_present(state['state_dict'], prefix="module.")
            align_and_update_state_dicts(model_state_dict, loaded_state_dict)
            model.load_state_dict(model_state_dict)

            scheduler = PolyLR(optimizer, max_iter=cfg.max_iter, power=0.9, last_step=curr_iter)
            optimizer.load_state_dict(state['optimizer'])

            if 'start_iter' in state:
                start_iter = state['start_iter']
            logger.info("=> loaded checkpoint '{}' (start_iter {})".format(checkpoint_fn, curr_iter))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))



    ##### dataset
    if cfg.dataset == 'scannetv2':
        if data_name == 'scannet':
            import data.scannetv2_inst
            dataset = data.scannetv2_inst.Dataset(start_iter=start_iter)
            dataset.trainLoader()
            # dataset.valLoader()
        #if data_name=="planteye":
        #    import data.planteye
            # dataset.valLoader()
        #if data_name=="planteye":
        #    import data.planteye
        else:
            print("Error: no data loader - " + data_name)
            exit(0)
    if data_name == 'planteye':
        import data.planteye_inst
        dataset = data.planteye_inst.Dataset(start_iter=start_iter)
        dataset.trainLoader()


    if start_iter < 0:
        start_iter = 0

    train(dataset.train_data_loader, model, model_fn, optimizer, start_iter=start_iter, scheduler=scheduler, save_to_disc=get_rank() == 0)

