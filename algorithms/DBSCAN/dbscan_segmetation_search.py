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
from scipy import stats
from util.log import logger


def read_labels(input_file):
    compressed_bin=open(input_file,"rb").read()
    binary_content=zlib.decompress(compressed_bin)
    lbl_array=array('B',binary_content)
    print(f"Length of label files is {len(lbl_array)}")
    lbl_array=np.array(lbl_array,np.compat.long)+1
    return lbl_array


def count_labels_group(inp):
    for i in np.unique(inp):
        cnts=(np.where(inp==i))
        print(f"Count of {i} label is {cnts[0].shape}")

def removez(pcd_points,pcd_colors,z):
    print(f"Number of Pointclouds are {pcd_points.shape}")
    #print(pcd_points[:,2].max())
    #print(pcd_points[:,2].min())
    mask=(pcd_points[:,2]>=z)
    pcd_points=pcd_points[mask]
    pcd_colors=pcd_colors[mask]
    print(f"Remaining Points are {pcd_points.shape}")
    return pcd_points,pcd_colors


def downsample(pcd,voxel_size=0.05):
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downpcd

def dbcluster(pcd,eps=2,min_points=1000):
    #with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        #labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    return labels

def remove_ground(pcd,z=100):
    pcd_points=np.asarray(pcd.points)
    pcd_colors=np.asarray(pcd.colors)
    pcd_points,pcd_colors=removez(pcd_points,pcd_colors,z=z)
    pcd_removed = o3d.geometry.PointCloud()
    pcd_removed.points = o3d.utility.Vector3dVector(pcd_points)
    pcd_removed.colors = o3d.utility.Vector3dVector(pcd_colors)
    return pcd_removed


def evaluate(pred,gt_labels,scene_name):
    # Set up the groud truth instances give all classes the label 1 and give them a different instance label consistent with Scannet evaluation
    target_instance=0
    target_label=1
    for i in np.unique(gt_labels):
        if (i==256):
            gt_labels[np.where(gt_labels==i)]=0
        else:
            target_instance+=1
            target=target_label*1000+target_instance
            gt_labels[np.where(gt_labels==i)]=target

    #print(np.unique(gt_labels))
    ## Setting up predications for comparisons
    masks=[]
    for i in np.unique(pred):
        if i==256 or i<0:
            continue
        pred_labels_cp=pred.copy()
        assert id(pred_labels_cp)!=id(pred)
        pred_labels_cp[np.where(pred_labels_cp!=i)]=0
        pred_labels_cp[np.where(pred_labels_cp==i)]=1
        masks.append(pred_labels_cp)
    #print(len(masks))
    try:
        masks=np.array(masks,dtype=np.int32)
    except:
        print("Too many points running out of memory, skipping!!")
        return 0
         
    # set all labels to 1
    label_id=np.ones(masks.shape[0],dtype=np.compat.long)
    # set all conf to one since model does not predict conf
    conf=np.ones(masks.shape[0],dtype=np.float64)
    pred_info = {}
    pred_info['conf'] = conf
    pred_info['label_id'] = label_id
    pred_info['mask'] =masks
    gt2pred, pred2gt = eval.assign_instances_for_scan(scene_name, pred_info, gt_pc=gt_labels)
    matches = {}
    matches[scene_name] = {}
    matches[scene_name]['gt'] = gt2pred
    matches[scene_name]['pred'] = pred2gt   
    ap_scores = eval.evaluate_matches(matches)
    #print(ap_scores.shape)
    avgs = eval.compute_averages(ap_scores)
    #if avgs["all_ap"]>0.99:
    #    print(f"Average AP is {avgs['all_ap']}")
    #    print(f"Masks Unique are {np.unique(masks)}")
    #    print(f"Masks Shaoe is {masks.shape}")
    #    print(f"Pred Unique are {np.unique(pred)}")
    #    print(f"Pred Shape is {pred.shape}")
    #    print(f"GT Unique are {np.unique(gt_labels)}")
    #    #exit()
    
    return avgs["all_ap"]




def grid_search(pcds,labels, hyperparams):
    best_estimator = None
    best_hyperparams = {}
    # hold best running score
    best_score = 0.0
    # get list of param values
    lists = hyperparams.values()
    # get all param combinations
    param_combinations = list(itertools.product(*lists))
    total_param_combinations = len(param_combinations)
    # iterate through param combinations
    for i, params in enumerate(param_combinations, 1):
        # fill param dict with params
        param_dict = {}
        for param_index, param_name in enumerate(hyperparams):
            param_dict[param_name] = params[param_index]
        # For all scans
        allap=[]
        for j in range(len(pcds)):
            pcd=pcds[j]
            label=labels[j].copy()
            # create estimator with specified params
            #estimator = clf(**param_dict)
            preds=dbcluster(pcd,**param_dict)
            assert label.shape==preds.shape,f"Shape of labels {labels.shape} Shape of predictions {preds.shape}"
            ap=evaluate(preds,label,str(i))
            allap.append(ap)

        allap=np.array(allap)
        estimator_score =np.mean(allap)
        logger.info(f"params {param_dict} AP {estimator_score}")

        print(f'[{i}/{total_param_combinations}] {param_dict}')
        print(f'Current mAP: {estimator_score}\n, Best mAP so far {best_score} \n Best Hyper Parameters So far are {best_hyperparams}')
        # if new high score, update high score
        # and best params 
        if estimator_score >= best_score:
                best_score = estimator_score
                best_hyperparams = param_dict


    return  best_hyperparams,best_score

def nearest_neighbour(originalpc,downpc,labels):
    print(np.array(originalpc.points).shape)
    print(np.array(downpc.points).shape)
    diff=np.array(originalpc.points)[:,None,:]-np.array(downpcd.points)
    print(diff.shape)


def display_labels(pcd,labels):
    #print(f"Print unique labels {np.unique(labels)}")
    max_label = sorted(np.unique(labels))[-1]
    if max_label==256:
        max_label = sorted(np.unique(labels))[-2]
    #colors = plt.get_cmap("tab20")(labels)
    colors = plt.get_cmap('RdYlBu')(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    colors[labels == 256] = 0
    #o3d.visualization.draw_geometries([pcd])
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])



def downsample_labels(pcd,labels,downpcd):
    """use KD Trees to downsample the labels using nearest neighbours"""
    pcd_colors=np.asarray(pcd.colors)
    pcd_colors[:,0]=labels
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    Nd=np.asfarray(downpcd.points).shape[0]
    out_label=np.zeros(Nd,dtype=np.long)
    for i in tqdm(range(Nd)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(downpcd.points[i], 10)
        l=np.asarray(pcd.colors)[idx[1:], 0]
        out_label[i]=np.long(scipy.stats.mode(l)[0])
    assert len(np.unique(labels))==len(np.unique(out_label))
    return out_label


def scaledPointcloud(pcd, scale):
    pcd_points = np.asarray(pcd.points) / scale
    pcd_colors = np.asarray(pcd.colors)
    res = o3d.geometry.PointCloud()
    res.points = o3d.utility.Vector3dVector(pcd_points)
    res.colors = o3d.utility.Vector3dVector(pcd_colors)
    return res



if __name__=="__main__":
    src_dir="/home/ec2-user/SageMaker/banglore_pc/banglore"
    label_dir="/home/ec2-user/SageMaker/banglore_annotation"
    scale = 100
    down_factor=0.03
    all_pcs=[]
    all_labels=[]
    for fs in os.listdir(label_dir):
        if fs.endswith(".zlib"):
            filename=os.path.join(src_dir,fs[:-4]+"ply")
            assert os.path.exists(filename),f"Point clouds donot exists {filename}"
            pcd=o3d.io.read_point_cloud(filename)
            print("original Points shape are")
            print(np.asarray(pcd.points).shape)
            pcd = scaledPointcloud(pcd, scale=scale)
            pcd_removed=remove_ground(pcd,z=1.0)
            downpcd=downsample(pcd_removed,down_factor)
            #print(f"The Downsample point clouds are {np.asarray(downpcd.points).shape}")
            gtfile=fs
            gt_file=os.path.join(label_dir,gtfile)
            assert os.path.exists(os.path.join(label_dir,gtfile)),f"Label file does not exists {gtfile}"
            gt_labels=read_labels(gt_file)
            print("original Labels shape are")
            print(gt_labels.shape)
            #display_labels(pcd,gt_labels)
            gt_down=downsample_labels(pcd,labels=gt_labels,downpcd=downpcd)
            #display_labels(downpcd,gt_down)
            all_pcs.append(downpcd)
            all_labels.append(gt_down)
            
            #break
    assert len(all_pcs)==len(all_labels), "Number of point clouds and labels donot match"
    hyperparams = {
        'eps': list(np.arange(0.001,0.6,0.001)),  
        'min_points': [5,10,15]
    }
    print(f"Hyperparameters are {hyperparams}")
    best_params,best_score=grid_search(all_pcs,all_labels,hyperparams=hyperparams)
    print(f"Best parameters for DB scan are {best_params} with score {best_score}")
            