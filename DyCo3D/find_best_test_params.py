import torch
import time
import numpy as np
import random
import os
import itertools
from tqdm import tqdm

from util.config import cfg
cfg.task = 'test'
from util.log import logger
import util.utils as utils
import util.eval as eval

import os.path as osp
from checkpoint import strip_prefix_if_present, align_and_update_state_dicts
from checkpoint import checkpoint

def init():
    global result_dir
    result_dir = os.path.join(cfg.exp_path, 'result', 'epoch{}_nmst{}_scoret{}_npointt{}'.format(cfg.test_epoch, cfg.TEST_NMS_THRESH, cfg.TEST_SCORE_THRESH, cfg.TEST_NPOINT_THRESH), cfg.split)
    backup_dir = os.path.join(result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'predicted_masks'), exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'gt_objs'), exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'pred_objs'), exist_ok=True)


    os.system('cp test.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    global semantic_label_idx
    #semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    semantic_label_idx = [0,1,2]

    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)


def test(model, model_fn, data_name, epoch,num_points,test_score,nms_thresh):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    if cfg.dataset == 'scannetv2':
        if data_name == 'scannet':
            from data.scannetv2_inst import Dataset
            dataset = Dataset(test=True)
            dataset.testLoader()
        else:
            print("Error: no data loader - " + data_name)
            exit(0)
    if data_name == 'planteye':
        import data.planteye_inst
        dataset = data.planteye_inst.Dataset(test=True)
        dataset.testLoader()

    dataloader = dataset.test_data_loader

    with torch.no_grad():
        model = model.eval()
        matches = {}
        for i, batch in enumerate(dataloader):
            N = batch['feats'].shape[0]
            if data_name == 'planteye':
                test_scene_name = dataset.test_file_names[int(batch['id'][0])].split('/')[-1][:-17]
            else:
                test_scene_name = dataset.test_file_names[int(batch['id'][0])].split('/')[-1][:12]


            preds = model_fn(batch, model, epoch)


            ##### get predictions (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)
            semantic_scores = preds['semantic']  # (N, nClass=20) float32, cuda
            semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda

            #print(semantic_pred.unique())

            pt_offsets = preds['pt_offsets']    # (N, 3), float32, cuda

            if (epoch > cfg.prepare_epochs):
                scores = preds['score']   # (nProposal, 1) float, cuda
                scores_pred = torch.sigmoid(scores.view(-1))

                proposals_idx, proposals_offset = preds['proposals']
                # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int, cpu
                proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int, device=scores_pred.device) # (nProposal, N), int, cuda
                proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

                semantic_id = torch.tensor(semantic_label_idx, device=scores_pred.device)[semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]] # (nProposal), long

                ##### score threshold
                score_mask = (scores_pred > test_score)
                scores_pred = scores_pred[score_mask]
                proposals_pred = proposals_pred[score_mask]
                semantic_id = semantic_id[score_mask]

                ##### npoint threshold
                proposals_pointnum = proposals_pred.sum(1)
                npoint_mask = (proposals_pointnum > num_points)
                scores_pred = scores_pred[npoint_mask]
                proposals_pred = proposals_pred[npoint_mask]
                semantic_id = semantic_id[npoint_mask]

                ##### nms
                if semantic_id.shape[0] == 0:
                    pick_idxs = np.empty(0)
                else:
                    proposals_pred_f = proposals_pred.float()  # (nProposal, N), float, cuda
                    intersection = torch.mm(proposals_pred_f, proposals_pred_f.t())  # (nProposal, nProposal), float, cuda
                    proposals_pointnum = proposals_pred_f.sum(1)  # (nProposal), float, cuda
                    proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
                    proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
                    cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
                    pick_idxs = non_max_suppression(cross_ious.cpu().numpy(), scores_pred.cpu().numpy(), nms_thresh)  # int, (nCluster, N)
                clusters = proposals_pred[pick_idxs]
                cluster_scores = scores_pred[pick_idxs]
                cluster_semantic_id = semantic_id[pick_idxs]

                nclusters = clusters.shape[0]

                ##### prepare for evaluation
                if cfg.eval:
                    pred_info = {}
                    pred_info['conf'] = cluster_scores.cpu().numpy()
                    pred_info['label_id'] = cluster_semantic_id.cpu().numpy()
                    pred_info['mask'] = clusters.cpu().numpy()
                    gt_file = os.path.join(cfg.data_root, cfg.dataset, cfg.split + '_gt', test_scene_name + '.txt')
                    gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, gt_file)
                    matches[test_scene_name] = {}
                    matches[test_scene_name]['gt'] = gt2pred
                    matches[test_scene_name]['pred'] = pred2gt


            ##### print
            logger.info("instance iter: {}/{} point_num: {} ncluster: {}".format(batch['id'][0] + 1, len(dataset.test_files), N, nclusters))
            torch.cuda.empty_cache()

        ##### evaluation
        if cfg.eval:
            ap_scores = eval.evaluate_matches(matches)
            avgs = eval.compute_averages(ap_scores)
            #eval.print_results(avgs)
        return avgs["all_ap"]


def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)



def find_params(model, model_fn, data_name,test_epoch):
    hyperparams = {
        "num_points": list(np.arange(5, 50,10)),
        "test_score": list(np.arange(0, 1,0.1)),
        "nms_thresh": list(np.arange(0, 1,0.05))
        #"num_points": [5],
        #"test_score": [0.6],
        #"nms_thresh": [0.25]
    }
    best_hyperparams = {}
    # hold best running score
    best_score = 0.0
    # get list of param values
    lists = hyperparams.values()
    # get all param combinations
    param_combinations = list(itertools.product(*lists))
    total_param_combinations = list(param_combinations)
    # iterate through param combinations
    for i, params in tqdm(enumerate(total_param_combinations, 1),total=len(total_param_combinations)):
        # fill param dict with params
        param_dict = {}
        for param_index, param_name in enumerate(hyperparams):
            param_dict[param_name] = params[param_index]
        # For all scans
        allap = []
        ap=test(model, model_fn, data_name,test_epoch,**param_dict)
        allap.append(ap)
        if ap >= best_score:
            best_score = ap
            best_hyperparams = param_dict
        print(f"Tried parameters {param_dict}")
        print(f"Current AP {ap}")
        print(f"Best parameters so far {best_hyperparams}")
        print(f"Best AP {best_score}")
    return best_hyperparams, best_score


if __name__ == '__main__':
    init()

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### model
    logger.info('=> creating model ...')
    logger.info('Classes: {}'.format(cfg.classes))

    #if model_name == 'pointgroup':
    from model.pointgroup.pointgroup import PointGroup as Network
    from model.pointgroup.pointgroup import model_fn_decorator
    #else:
    #    print("Error: no model version " + model_name)
    #    exit(0)
    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### model_fn (criterion)
    model_fn = model_fn_decorator(test=True)

    checkpoint_fn = cfg.resume
    if osp.isfile(checkpoint_fn):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_fn))
        state = torch.load(checkpoint_fn)
        curr_iter = state['iteration'] + 1
        model_state_dict = model.state_dict()
        loaded_state_dict = strip_prefix_if_present(state['state_dict'], prefix="module.")
        align_and_update_state_dicts(model_state_dict, loaded_state_dict)
        model.load_state_dict(model_state_dict)
        # model.load_state_dict(state['state_dict'])

        logger.info("=> loaded checkpoint '{}' (start_iter {})".format(checkpoint_fn, curr_iter))
    else:
        raise RuntimeError


    ##### evaluate
    find_params(model, model_fn, data_name, cfg.test_epoch)
    #test(model, model_fn, data_name, cfg.test_epoch)
