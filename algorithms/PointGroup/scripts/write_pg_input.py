import numpy as np
from numpy.core.fromnumeric import mean
import open3d as o3d
import os
import matplotlib.pyplot as plt
import zlib
from array import array
import scipy
#import util.eval_planteye as eval
from tqdm import tqdm
import argparse
#from util.log import logger
from utils import display_labels

args=argparse.ArgumentParser()
args.add_argument("--src_dir", type=str, default="/media/asad/ADAS_CV/pointclouds-Instance-Segmentation/data/raw_data/pointclouds")
args.add_argument("--label_dir", type=str, default="/media/asad/ADAS_CV/pointclouds-Instance-Segmentation/data/raw_data/annotation")
args.add_argument("--dst_dir", type=str, default="/media/asad/ADAS_CV/pointclouds-Instance-Segmentation/data/Dyco3D/train")
args.add_argument("--dst_label_dir", type=str, default="/media/asad/ADAS_CV/pointclouds-Instance-Segmentation/data/Dyco3D/train")
args = args.parse_args()

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
        if i == 256:
            gt_labels[np.where(gt_labels == i)] = 1000
        else:
            target = target_label * 1000 + target_instance
            # print(i,target)
            gt_labels[np.where(gt_labels == i)] = target
            target_instance += 1
    # print(np.unique(gt_labels))
    print(f"Unique labels after modifications {np.unique(gt_labels)}")
    out_dir = os.path.join(label_dir, gt_file.split("/")[-1][:-5] + ".txt")
    np.savetxt(out_dir, gt_labels, delimiter="\n", fmt="%u")


def count_labels_group(inp):
    for i in np.unique(inp):
        cnts = np.where(inp == i)
        print(f"Count of {i} label is {cnts[0].shape}")


def removez(pcd_points, pcd_colors, z):
    print(f"Number of Pointclouds are {pcd_points.shape}")
    # print(pcd_points[:,2].max())
    # print(pcd_points[:,2].min())
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


def dbcluster(pcd, eps=2, min_points=1000):
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


def nearest_neighbour(originalpc, downpc, labels):
    print(np.array(originalpc.points).shape)
    print(np.array(downpc.points).shape)
    diff = np.array(originalpc.points)[:, None, :] - np.array(downpcd.points)
    print(diff.shape)


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
    src_dir = args.src_dir
    label_dir = args.label_dir
    dst_src= args.dst_dir
    dst_labels=args.dst_label_dir
    scale = 100
    #down_factor = voxel grid size
    down_factor=0.03
    all_pcs = []
    all_labels = []
    for fs in tqdm(os.listdir(label_dir)):
        gtfile = os.path.join(label_dir, fs)
        plyfile = os.path.join(src_dir, fs[:-5] + ".ply")
        pcd = o3d.io.read_point_cloud(plyfile)
        print(f"Max of Original clouds are {np.max(np.asarray(pcd.points))}")
        print(f"Min of Original point clouds are {np.min(np.asarray(pcd.points))}")
        pcd = scaledPointcloud(pcd, scale=scale)
        assert os.path.exists(
            os.path.join(label_dir, gtfile)
        ), f"Label file does not exists {gtfile}"
        # print(f"original point clouds are {np.asarray(pcd.points).shape}")
        # o3d.visualization.draw_geometries([pcd])
        pcd_removed = remove_ground(pcd, z=0.90)
        print(f"Shape of removed {np.asarray(pcd_removed.points).shape}")
        print(f"Max of removed point clouds are {np.max(np.asarray(pcd_removed.points))}")
        print(f"Min of removed point clouds are {np.min(np.asarray(pcd_removed.points))}")
        # o3d.visualization.draw_geometries([pcd_removed])
        downpcd = downsample(pcd_removed, down_factor)
        #downpcd=pcd_removed.uniform_down_sample(4)
        print(f"The Downsample point clouds are {np.asarray(downpcd.points).shape}")
        o3d.io.write_point_cloud(
            os.path.join(dst_src, fs[:-5] + ".ply"),
            downpcd,
            write_ascii=False,
            compressed=False,
            print_progress=False,
        )
        gt_labels = read_labels(gtfile)
        #display_labels(pcd,gt_labels)
        gt_down = downsample_labels(pcd, labels=gt_labels, downpcd=downpcd)
        write_labels(gt_down, dst_labels, fs)
        #display_labels(downpcd,gt_down)
