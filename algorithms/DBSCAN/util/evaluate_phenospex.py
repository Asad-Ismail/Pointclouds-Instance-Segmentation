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


src_dir="/media/asad/8800F79D00F79104/banglore_ply/1_label_phenospex_test"
label_dir="/media/asad/8800F79D00F79104/banglore_ply/1_label_planteye_test"

for fs in os.listdir(src_dir):
        if fs.endswith(".ply"):
            filename=os.path.join(src_dir,fs)
            pcd=o3d.io.read_point_cloud(filename)
            o3d.visualization.draw_geometries([pcd])
            print(f"original point clouds are {np.asarray(pcd.points).shape}")
            points_list=[]
            color_list=[]
            for fl in os.listdir(label_dir):
                labelname=os.path.join(label_dir,fl)
                label=o3d.io.read_point_cloud(filename)
                o3d.visualization.draw_geometries([label])
                points_list.append(np.asarray(label.points))
                color_list.append(np.asarray(label.colors))
            allpoints=np.asarray(points_list)
            allcolors=np.asarray(color_list)
            

