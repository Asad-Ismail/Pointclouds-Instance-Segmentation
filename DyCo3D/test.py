import torch
import time
import numpy as np
import random
import os

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


def test(model, model_fn, data_name, epoch):
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
                score_mask = (scores_pred > cfg.TEST_SCORE_THRESH)
                scores_pred = scores_pred[score_mask]
                proposals_pred = proposals_pred[score_mask]
                semantic_id = semantic_id[score_mask]

                ##### npoint threshold
                proposals_pointnum = proposals_pred.sum(1)
                npoint_mask = (proposals_pointnum > cfg.TEST_NPOINT_THRESH)
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
                    pick_idxs = non_max_suppression(cross_ious.cpu().numpy(), scores_pred.cpu().numpy(), cfg.TEST_NMS_THRESH)  # int, (nCluster, N)
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

            ##### save files
            if cfg.save_semantic:
                os.makedirs(os.path.join(result_dir, 'semantic'), exist_ok=True)
                semantic_np = semantic_pred.cpu().numpy()
                np.save(os.path.join(result_dir, 'semantic', test_scene_name + '.npy'), semantic_np)

            if cfg.save_pt_offsets:
                os.makedirs(os.path.join(result_dir, 'coords_offsets'), exist_ok=True)
                pt_offsets_np = pt_offsets.cpu().numpy()
                coords_np = batch['locs_float'].numpy()
                coords_offsets = np.concatenate((coords_np, pt_offsets_np), 1)   # (N, 6)
                np.save(os.path.join(result_dir, 'coords_offsets', test_scene_name + '.npy'), coords_offsets)


            if(epoch > cfg.prepare_epochs and cfg.save_instance):
                f = open(os.path.join(result_dir, test_scene_name + '.txt'), 'w')
                for proposal_id in range(nclusters):
                    clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                    semantic_label = np.argmax(np.bincount(semantic_pred[np.where(clusters_i == 1)[0]].cpu()))
                    score = cluster_scores[proposal_id]
                    f.write('predicted_masks/{}_{:03d}.txt {} {:.4f}'.format(test_scene_name, proposal_id, semantic_label_idx[semantic_label], score))
                    if proposal_id < nclusters - 1:
                        f.write('\n')
                    np.savetxt(os.path.join(result_dir, 'predicted_masks', test_scene_name + '_%03d.txt' % (proposal_id)), clusters_i, fmt='%d')
                f.close()



            if cfg.save_instance:
                from util.draw_utils import write_ply_rgb, write_ply_color
                xyz, rgb, label, instance_label = torch.load(os.path.join(cfg.data_root, cfg.dataset, cfg.split, test_scene_name + '_inst_nostuff.pth'))
                max_ins_label = np.max(instance_label)
                instance_label[instance_label<0] = max_ins_label + 1
                label[label < 0] = 20
                write_ply_color(xyz, instance_label, os.path.join(result_dir, 'gt_objs/{}_gt.obj'.format(test_scene_name)))
                pred_ins_labels = np.zeros_like(label)
                num = 1
                for proposal_id in range(nclusters):
                    clusters_i = clusters[proposal_id].cpu().numpy()
                    pred_ins_labels[np.where(clusters_i == 1)[0]] = num
                    num += 1

                write_ply_color(xyz, pred_ins_labels, os.path.join(result_dir, 'pred_objs/{}_pred.obj'.format(test_scene_name)))


            ##### print
            logger.info("instance iter: {}/{} point_num: {} ncluster: {}".format(batch['id'][0] + 1, len(dataset.test_files), N, nclusters))
            torch.cuda.empty_cache()

        ##### evaluation
        if cfg.eval:
            ap_scores = eval.evaluate_matches(matches)
            avgs = eval.compute_averages(ap_scores)
            eval.print_results(avgs)


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

    ##### load model
    # utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, cfg.test_epoch, dist=False, f=cfg.pretrain)      # resume from the latest epoch, or specify the epoch to restore
    # checkpoint_fn = cfg.resume
    # if osp.isfile(checkpoint_fn):
    #     checkpoint_fn = cfg.resume
    #     if osp.isfile(checkpoint_fn):
    #         logger.info("=> loading checkpoint '{}'".format(checkpoint_fn))
    #         state = torch.load(checkpoint_fn)
    #         curr_iter = state['iteration'] + 1
    #         start_iter = curr_iter
    #         model_state_dict = model.state_dict()
    #         loaded_state_dict = strip_prefix_if_present(state['state_dict'], prefix="module.")
    #         align_and_update_state_dicts(model_state_dict, loaded_state_dict)
    #         model.load_state_dict(model_state_dict)
    #         # model.load_state_dict(state['state_dict'])
    #         if 'start_iter' in state:
    #             start_iter = state['start_iter']
    #         logger.info("=> loaded checkpoint '{}' (start_iter {})".format(checkpoint_fn, curr_iter))
    # else:
    #     raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))

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
    test(model, model_fn, data_name, cfg.test_epoch)
