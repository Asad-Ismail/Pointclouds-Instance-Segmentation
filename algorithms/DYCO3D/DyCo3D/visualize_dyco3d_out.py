import numpy as np
from numpy.core.defchararray import asarray
from numpy.lib.type_check import asfarray
import open3d as o3d
import argparse
import os
import sys
import matplotlib.pyplot as plt
import zlib
from array import array
import json


def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(-1.0,0.0,0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)
def custom_draw_geometry_with_custom_fov(pcd, fov_step):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    ctr.change_field_of_view(step=fov_step)
    print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
    vis.run()
    vis.destroy_window()

def visualize(
    src_dir="",
    label_dir="",
    pred_dir="",
):
    for mfile in os.listdir(src_dir):
        if not mfile.endswith(".ply"):
            continue
        l_path = os.path.join(label_dir, mfile[:-4] + ".txt")
        f_path = os.path.join(src_dir, mfile)
        #print(f"The file path is {f_path}")
        #print(f"The label file path is {l_path}")
        pcd = o3d.io.read_point_cloud(f_path)
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors)
        if l_path.endswith(".txt"):
            with open(l_path, "r") as pc:
                pcd = pc.readlines()
                pcd_labels_class = [
                    int((float(iteminter)) // 1000)
                    for item in pcd
                    for iteminter in item.strip().split()
                ]
                pcd_labels_instance = [
                    int(float(iteminter)) % 1000
                    for item in pcd
                    for iteminter in item.strip().split()
                ]

        #pcd_labels_colors = np.asarray(pcd_labels_class)
        pcd_labels_instance = np.asarray(pcd_labels_instance)
        #classes = np.unique(pcd_labels_class)
        instances = np.unique(pcd_labels_instance)  
        
        pcd_l_colors = np.asarray(pcd_colors).copy()
        pcd_pred_colors = np.asarray(pcd_colors).copy()

        for i in instances:
            color = [float(np.random.rand(1)) for _ in range(3)]
            if i == 0:
                continue
                #pcd_l_colors[np.where(pcd_labels_instance == i)] = [0, 0, 0]
            pcd_l_colors[np.where(pcd_labels_instance == i)] = color
        # Pred files
        for pfile in os.listdir(pred_dir):
            if mfile[:-4] in pfile:
                f_pred = os.path.join(pred_dir, pfile)
                if f_pred.endswith(".txt"):
                    with open(f_pred, "r") as pc:
                        pred_pcd = pc.readlines()
                        pcd_pred_class = [
                            int((float(iteminter)))
                            for item in pred_pcd
                            for iteminter in item.strip().split()
                        ]
                        pcd_pred_class = np.asfarray(pcd_pred_class).astype(np.int32)
                        # print(np.unique(pcd_pred_class))
                        color = [float(np.random.rand(1)) for _ in range(3)]
                        pcd_pred_colors[np.where(pcd_pred_class != 0)] = color
                        #pcd_pred_colors[np.where(pcd_pred_class == 0)] = [0, 0, 0]
        pcd_pred = o3d.geometry.PointCloud()
        pcd_label = o3d.geometry.PointCloud()
        pcd_org = o3d.geometry.PointCloud()
        # shift points to display on the same image
        pcd_label.points = o3d.utility.Vector3dVector(pcd_points)
        pcd_label.colors = o3d.utility.Vector3dVector(pcd_l_colors)
        pcd_pred.points = o3d.utility.Vector3dVector(pcd_points)
        pcd_pred.colors = o3d.utility.Vector3dVector(pcd_pred_colors)
        pcd_org.points = o3d.utility.Vector3dVector(pcd_points)
        pcd_org.colors = o3d.utility.Vector3dVector(pcd_colors)
        
        offset=10
        offset_labels=pcd_points.copy()
        offset_labels[:,0]+=offset
        offset_pred=pcd_points.copy()
        offset_pred[:,0]+=offset*2
        comb_points=np.concatenate((pcd_points,offset_labels,offset_pred))
        comb_colors=np.concatenate((pcd_colors,pcd_l_colors,pcd_pred_colors))
        pcd_comb = o3d.geometry.PointCloud()
        pcd_comb.points = o3d.utility.Vector3dVector(comb_points)
        pcd_comb.colors = o3d.utility.Vector3dVector(comb_colors)
        
        
        #o3d.visualization.draw_geometries([pcd_org])
        #o3d.visualization.draw_geometries([pcd_label])
        #o3d.visualization.draw_geometries([pcd_pred])
        #o3d.visualization.draw_geometries([pcd_comb])
        custom_draw_geometry_with_custom_fov(pcd_comb, 90.0)
        #custom_draw_geometry_with_custom_fov(pcd_comb, -90.0)
        custom_draw_geometry_with_rotation(pcd_comb)


if __name__ == "__main__":
    src_dir = "dataset/planteye/val"
    label_dir = "dataset/planteye/val"
    pred_dir = "exp/planteye/pointgroup/pointgroup_default_planteye/result/epoch10000_nmst0.3_scoret0.5_npointt20/val/predicted_masks"
    visualize(src_dir, label_dir, pred_dir)
