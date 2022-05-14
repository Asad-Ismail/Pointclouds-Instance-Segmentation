import numpy as np
from numpy.core.fromnumeric import mean
import open3d as o3d
import argparse
import os
import matplotlib.pyplot as plt
import zlib
from array import array
import scipy
import util.eval_planteye as eval
from tqdm import tqdm
import itertools
import multiprocessing
from util.log import logger
from utils import display_labels


def read_labels(input_file):
    compressed_bin = open(input_file, "rb").read()
    binary_content = zlib.decompress(compressed_bin)
    lbl_array = array("B", binary_content)
    print(f"Length of label files is {len(lbl_array)}")
    lbl_array = np.array(lbl_array, np.compat.long) + 1
    return lbl_array


def write_labels(gt_labels, label_dir, gt_file):
    target_instance = 1
    target_label = 2
    # print(np.unique(gt_labels))
    for i in np.unique(gt_labels):
        target_instance += 1
        if i == 256:
            gt_labels[np.where(gt_labels == i)] = 1000
        else:
            target = target_label * 1000 + target_instance
            # print(i,target)
            gt_labels[np.where(gt_labels == i)] = target
    # print(np.unique(gt_labels))
    out_dir = os.path.join(label_dir, gt_file.split("/")[-1][:-5] + ".txt")
    np.savetxt(out_dir, gt_labels, delimiter="\n", fmt="%u")


def count_labels_group(inp):
    for i in np.unique(inp):
        cnts = np.where(inp == i)
        print(f"Count of {i} label is {cnts[0].shape}")


def removez(pcd_points, pcd_colors, z):
    print(f"Number of Pointclouds are {pcd_points.shape}")
    mask = pcd_points[:, 2] >= z
    pcd_points = pcd_points[mask]
    pcd_colors = pcd_colors[mask]
    print(f"Remaining Points are {pcd_points.shape}")
    return pcd_points, pcd_colors


def scaledPointcloud(pcd, scale):
    pcd_points = np.asarray(pcd.points) / scale
    pcd_colors = np.asarray(pcd.colors)
    res = o3d.geometry.PointCloud()
    res.points = o3d.utility.Vector3dVector(pcd_points)
    res.colors = o3d.utility.Vector3dVector(pcd_colors)
    return res


def downsample(pcd, voxel_size=0.05):
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downpcd


def dbcluster(pcd, eps=2, min_points=10):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
        )
    return labels


def remove_ground(pcd, z=100):
    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors)
    pcd_points, pcd_colors = removez(pcd_points, pcd_colors, z=z)
    pcd_removed = o3d.geometry.PointCloud()
    pcd_removed.points = o3d.utility.Vector3dVector(pcd_points)
    pcd_removed.colors = o3d.utility.Vector3dVector(pcd_colors)
    return pcd_removed


def evaluate(pred, gt_labels, scene_name):
    # Set up the groud truth instances give all classes the label 1 and give them a different instance label consistent with Scannet evaluation
    target_instance = 1
    target_label = 1
    for i in np.unique(gt_labels):
        target_instance += 1
        if i == 256:
            gt_labels[np.where(gt_labels == i)] = 0
        else:
            target = target_label * 1000 + target_instance
            # print(i,target)
            gt_labels[np.where(gt_labels == i)] = target

    # print(np.unique(gt_labels))
    ## Setting up predications for comparisons
    masks = []
    for i in np.unique(pred):
        if i >= 255 or i < 0:
            continue
        pred_labels_cp = pred.copy()
        assert id(pred_labels_cp) != id(pred)
        pred_labels_cp[np.where(pred_labels_cp != i)] = 0
        pred_labels_cp[np.where(pred_labels_cp == i)] = 1
        masks.append(pred_labels_cp)
    # print(len(masks))
    masks = np.array(masks, dtype=np.int32)
    # set all labels to 1
    label_id = np.ones(masks.shape[0], dtype=np.compat.long)
    # set all conf to one since model does not predict conf
    conf = np.ones(masks.shape[0], dtype=np.float64)
    # print(masks.shape)
    pred_info = {}
    pred_info["conf"] = conf
    pred_info["label_id"] = label_id
    pred_info["mask"] = masks
    gt2pred, pred2gt = eval.assign_instances_for_scan(
        scene_name, pred_info, gt_pc=gt_labels
    )
    matches = {}
    matches[scene_name] = {}
    matches[scene_name]["gt"] = gt2pred
    matches[scene_name]["pred"] = pred2gt
    ap_scores = eval.evaluate_matches(matches)
    # print(ap_scores.shape)
    avgs = eval.compute_averages(ap_scores)
    # print(avgs)
    return avgs["all_ap"]


def dbscan_segmenation(pcds, labels, hyperparams, vis=False):
    allap = []
    for j in range(len(pcds)):
        pcd = pcds[j]
        label = labels[j]
        # create estimator with specified params
        # estimator = clf(**param_dict)
        preds = dbcluster(pcd, **hyperparams)
        assert (
            label.shape == preds.shape
        ), f"Shape of labels {labels.shape} Shape of predictions {preds.shape}"
        ap = evaluate(preds, label, str(j))
        if vis:
            #display_labels(pcd, label)
            display_labels(pcd, preds)
        allap.append(ap)

    allap = np.array(allap)
    ap = np.mean(allap)
    return ap


def nearest_neighbour(originalpc, downpc, labels):
    print(np.array(originalpc.points).shape)
    print(np.array(downpc.points).shape)
    diff = np.array(originalpc.points)[:, None, :] - np.array(downpcd.points)
    print(diff.shape)


def display_labels(pcd, labels):
    # print(f"Print unique labels {np.unique(labels)}")
    if (labels > 1000).any():
        labels = labels % 1000
    print(f"Unique labels are {np.unique(labels)}")
    max_label = sorted(np.unique(labels))[-1]

    if max_label <= 255:
        max_label = sorted(np.unique(labels))[-2]
    pcd_colors = np.asarray(pcd.colors)
    pcd_l_colors = np.asarray(pcd_colors).copy()

    unique_labels = np.unique(labels)
    for i in np.unique(unique_labels):
            color = [float(np.random.rand(1)) for _ in range(3)]
            if i == 0:
                continue
                #pcd_l_colors[np.where(pcd_labels_instance == i)] = [0, 0, 0]
            pcd_l_colors[np.where(labels == i)] = color
    
    pcd.colors = o3d.utility.Vector3dVector(pcd_l_colors)
    o3d.visualization.draw_geometries([pcd])


def downsample_labels(pcd, labels, downpcd):
    """use KD Trees to downsample the labels using nearest neighbours"""
    pcd_colors = np.asarray(pcd.colors)
    pcd_colors[:, 0] = labels
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    Nd = np.asfarray(downpcd.points).shape[0]
    out_label = np.zeros(Nd, dtype=np.long)
    for i in tqdm(range(Nd)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(downpcd.points[i], 10)
        l = np.asarray(pcd.colors)[idx[1:], 0]
        out_label[i] = np.long(scipy.stats.mode(l)[0])
    assert len(np.unique(labels)) == len(np.unique(out_label))
    return out_label


if __name__ == "__main__":
    src_dir = "/media/asad/ADAS_CV/pointclouds/planteye_allscans/merge/banglore"
    label_dir = "/media/asad/ADAS_CV/pointclouds/banglore_annotation"
    # down_factor=4
    scale = 100
    down_factor=0.03
    all_pcs = []
    all_labels = []
    for fs in os.listdir(label_dir):
        gtfile = os.path.join(label_dir, fs)
        plyfile = os.path.join(src_dir, fs[:-5] + ".ply")
        pcd = o3d.io.read_point_cloud(plyfile)
        pcd = scaledPointcloud(pcd, scale=scale)
        assert os.path.exists(os.path.join(label_dir, gtfile)), f"Label file does not exists {gtfile}"
        pcd_removed = remove_ground(pcd, z=1.000)
        # print(f"Removed Ground point clouds are {np.asarray(pcd_removed.points).shape}")
        # o3d.visualization.draw_geometries([pcd_removed])
        downpcd = downsample(pcd_removed, down_factor)
        print(f"The Downsample point clouds are {np.asarray(downpcd.points).shape}")
        # o3d.visualization.draw_geometries([downpcd])
        # continue
        gt_labels = read_labels(gtfile)
        # if (np.asarray(pcd.points).shape[0]==gt_labels.shape[0]):
        #    continue
        # display_labels(pcd,gt_labels)
        gt_down = downsample_labels(pcd, labels=gt_labels, downpcd=downpcd)
        #write_labels(gt_down, "/media/asad/ADAS_CV/pointclouds/ds30d/labels", fs)
        # display_labels(downpcd,gt_down)
        all_pcs.append(downpcd)
        all_labels.append(gt_down)
        assert len(all_pcs) == len(all_labels), "Number of point clouds and labels donot match"
        hyperparams = {"eps": 0.097, "min_points": 5}
        dbscan_segmenation(all_pcs, all_labels, hyperparams, vis=True)
