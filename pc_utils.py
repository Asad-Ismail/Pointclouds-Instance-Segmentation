import numpy as np
import open3d as o3d
import argparse
import os
import sys
import matplotlib.pyplot as plt
import zlib
from array import array
import json


args=argparse.ArgumentParser(description="Inputs Args")
args.add_argument("--src_dir",type=str,default="/media/asad/adas_cv_2/pointcloudlibs/raw_points")
args.add_argument("--dst_dir",type=str,default="/media/asad/adas_cv_2/pointcloudlibs/sagemaker")
args.add_argument("--downsample",type=int,default=None)
args.add_argument("--debug",type=bool,default=True)
args.add_argument("--task",type=str,default="sagemaker",choices=["merge","plytopcd","sagemaker","validate","vis"])
args.add_argument("--SM-label",type=str,default="s3://sagemaker-us-east-1-470086202700/labelling-pointcloud/pc_labelling_merge")
args.add_argument("--label_dir",type=str,default="")
arg=args.parse_args()



colors=np.array([   
      [120, 120, 120],
      [204, 5, 255],
      [230, 230, 230],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255]],dtype=np.float32)
colors/=255.0
print(f"The length of labels color shape is {len(colors)}")


def read_file(input_file):
    with open(input_file,"r") as f:
        line = f.readline()
        cnt = 1
        while line:
            print("Line {}: {}".format(cnt, line.strip()))
            line = f.readline()
            cnt += 1
            if (cnt>11):
                label=int(line.split(" ")[4])
                obj=int(line.split(" ")[5])



def read_labels(input_file):
    compressed_bin=open(input_file,"rb").read()
    binary_content=zlib.decompress(compressed_bin)
    lbl_array=array('B',binary_content)
    print(f"Length of label files is {len(lbl_array)}")
    return lbl_array
    



def write_point_cloud_to_text(src_dir,dst_dir,down_sample,sagemaker_dst_dir):
    """
    Reads all point cloud files from a diectory and write a text file in the format that allows amazon sage maker groudth truth useable
    outfile has row x,y,z,r,g,b (xyz are float r,g,b are int)
    """
    os.makedirs(dst_dir,exist_ok=True)
    for i,f in enumerate(os.listdir(src_dir)): 
        f_path=os.path.join(src_dir,f)
        d_f=f.split(".")[0]+".txt"
        d_path=os.path.join(dst_dir,d_f)
        pcd = o3d.io.read_point_cloud(f_path)
        if down_sample is not None:
            print (f"Downsampling data points by factor of {down_sample}")
            pcd = pcd.voxel_down_sample(voxel_size=down_sample)
        pcd_point=np.asarray(pcd.points)
        #print(pcd_point[:,0].max())
        #print(pcd_point[:,0].min())
        print(f"Number of point clouds {pcd_point.shape}")
        print(f"Max Point cloud x {max(pcd_point[0])}, Min Point cloud x {min(pcd_point[0])}")
        print(f"Max Point cloud y {max(pcd_point[1])}, Min Point cloud y {min(pcd_point[1])}")
        print(f"Max Point cloud z {max(pcd_point[2])}, Min Point cloud z {min(pcd_point[2])}")
        pcd_colors=(np.asarray(pcd.colors)*255).astype(np.uint8)
        print(f"Writing output text file {d_path}")

        sagemaker_prefix="s3://sagemaker-us-east-1-470086202700/labelling-pointcloud"
        manifest_path=os.path.join(sagemaker_dst_dir,d_f)
        data={}
        data["source-ref"]=manifest_path
        data["source-ref-metadata"]={"format": "text/xyzrgb","unix-timestamp":1566861644.759115+i}
        with open(dst_dir+"/files.manifest","a") as f:
            res=json.dumps(data)
            f.write(res)
            f.write("\n")


        with open(d_path,"w") as f:
            for point,color in zip(pcd_point,pcd_colors):
                out=str(point[0])+" "+str(point[1])+" "+str(point[2])+" "+str(color[0])+" "+str(color[1])+" "+str(color[2])
                f.writelines(out)
                f.write("\n")


             

def convert_to_pcd(src_dir,dst_dir,down_sample):
    """
    Convert the files given in srx dir to pcd and write results in dst_dir
    """
    for i,f in enumerate(os.listdir(src_dir)): 
        f_path=os.path.join(src_dir,f)
        d_f=f.split(".")[0]+".pcd"
        d_path=os.path.join(dst_dir,d_f)
        pcd = o3d.io.read_point_cloud(f_path)
        if down_sample is not None:
            print (f"Downsampling data points by factor of {down_sample}")
            pcd = pcd.voxel_down_sample(voxel_size=down_sample)
        print(f"Processed {i+1} files; Current processed {f} having points {np.asarray(pcd.points).shape}") 
        o3d.io.write_point_cloud(d_path, pcd,write_ascii=True)
        if arg.debug:
            o3d.visualization.draw_geometries([pcd])

def joinstring(string_list):
    prefix=string_list[0]
    out=""
    for i in range(1,len(string_list)):
        out+="_"+str(string_list[i])
    out=prefix+out
    return out

def merge_pcd_files(src_dir,dst_dir,down_sample):
    """
    Merge Two point cloud files by appending the points of both xyz and colors and writing the resulting file in dst dir
    Assumes the Merged file has "M" and "S" files
    """
    for i,f in enumerate(os.listdir(src_dir)):
        split_f=f.split("_")
        if split_f[2] != "M":
            continue
        fm_path=os.path.join(src_dir,f)
        fs_file=split_f
        dst_file=fs_file
        dst_file[2]="O"
        dst_file=joinstring(dst_file)
        dst_path=os.path.join(dst_dir,dst_file)
        fs_file[2]="S"
        fs_file=joinstring(fs_file)
        fs_path=os.path.join(src_dir,fs_file)
        # check that the file exists
        assert os.path.exists(fs_path), f"File {fs_path} does not exist"
        assert os.path.exists(fm_path),f"File {fm_path} does not exist"
        pcd_s = o3d.io.read_point_cloud(fs_path)
        pcd_m = o3d.io.read_point_cloud(fm_path)
        
        pcd_m_points=np.asarray(pcd_m.points)
        pcd_s_points=np.asarray(pcd_s.points)
        pcd_points=np.concatenate([pcd_m_points,pcd_s_points],axis=0)

        print(pcd_points[:,2].max())
        print(pcd_points[:,2].min())
        #filtered_pcd_point_index=np.where(pcd_point[:,2]>100)[0]
        #filtered_pcd_points=pcd_point[filtered_pcd_point_index,:]


        pcd_m_colors=np.asarray(pcd_m.colors)
        pcd_s_colors=np.asarray(pcd_s.colors)
        pcd_colors=np.concatenate([pcd_m_colors,pcd_s_colors],axis=0)
        #filtered_pcd_colors=pcd_colors[filtered_pcd_point_index,:]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

        if down_sample is not None:
            print (f"Downsampling data points by factor of {down_sample}")
            pcd = pcd.voxel_down_sample(voxel_size=down_sample)
        print(f"Processed {i+1} files; {fs_file} has points {np.asarray(pcd_s.points).shape}") 
        print(f"Processed {i+1} files; {f} has points {np.asarray(pcd_m.points).shape}") 
        print(f"Combined {i+1} files; Combined file {dst_file} has points {np.asarray(pcd.points).shape}") 

        if arg.debug:
            o3d.visualization.draw_geometries([pcd_s])
            o3d.visualization.draw_geometries([pcd_m])
            o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(os.path.join(dst_dir,f), pcd, write_ascii=False, compressed=False, print_progress=False)             


def validate(src_dir,label_dir):
    for mfile in os.listdir(src_dir):
        f_path=os.path.join(src_dir,mfile)
        l_path=os.path.join(label_dir,mfile[:-4]+".zlib")
        print(f"The file path is {f_path}")
        print(f"The label file path is {l_path}")
        with open(f_path,"r") as pc:
            pcd=pc.readlines()
            pcd_points=[[float(iteminter) for iteminter in item.strip().split()[:3]] for item in pcd]
            pcd_colors=[[float(iteminter)/255.0 for iteminter in item.strip().split()[3:]]  for item in pcd]
            pcd_m_points=np.asarray(pcd_points)
            pcd_m_colors=np.asarray(pcd_colors)
            pcd_org_colors=pcd_m_colors.copy()
            pcd_org_points=pcd_m_points.copy()
            # Shift X point by 1 m to descriminate between orignal and labelled
            pcd_org_points[:,2]+=20
            print(f"shape of single dimension is {pcd_org_points[:]}")
            #Concatenate original points and labelled points for vis
            print(f"Number of points in pointcloud files are {pcd_m_points.shape}")
            arr=read_labels(l_path)
            arr=np.array(arr)
            unique_labels=np.unique(arr)
            print(f"Number of unique labels in {mfile} are {len(unique_labels)}")
            print(f"Unique labels are {unique_labels}")
            for index,i in enumerate(unique_labels):
                #color=[float(np.random.rand(1)) for _ in range(3)]
                if (i==255):
                    pcd_m_colors[np.where(arr==i)]=[0,0,0]
                else:
                    pcd_m_colors[np.where(arr==i)]=colors[index]
            print(pcd_org_points.shape)
            pcd_org_points=np.concatenate((pcd_org_points,pcd_m_points),axis=0) 
            pcd_org_colors=np.concatenate((pcd_org_colors,pcd_m_colors),axis=0)
            print(pcd_org_points.shape) 
            pcd_label = o3d.geometry.PointCloud()
            pcd_org = o3d.geometry.PointCloud()
            pcd_label.points = o3d.utility.Vector3dVector(pcd_m_points)
            pcd_label.colors = o3d.utility.Vector3dVector(pcd_m_colors)
            pcd_org.points=o3d.utility.Vector3dVector(pcd_org_points)
            pcd_org.colors=o3d.utility.Vector3dVector(pcd_org_colors)
            o3d.visualization.draw_geometries([pcd_org])




def vispointclouds(src_dir):
    for fs in os.listdir(src_dir):
        filename=os.path.join(src_dir,fs)
        pcd=o3d.io.read_point_cloud(filename)
        o3d.visualization.draw_geometries([pcd])
        print(f"Processed has points {np.asarray(pcd.points).shape}") 



if  __name__=="__main__":
    #test()
    print(f"Passed Arguments are {len( vars(arg) )}")
    if (len( vars(arg))<3):
        print(f"usage is python pcl_to_pcd.py --src_dir PATH_TO_SRC_DIR --dst_dir PATH_TO_DEST_DIR --downsample downsampling param")
        exit(-1)
    print(arg.task)
    if arg.task=="plytopcd":
        convert_to_pcd(arg.src_dir,arg.dst_dir,arg.downsample)  
    elif arg.task=="sagemaker":   
        write_point_cloud_to_text(arg.src_dir,arg.dst_dir,arg.downsample,arg.SM-label)
    elif arg.task=="merge":
        merge_pcd_files(arg.src_dir, arg.dst_dir, arg.downsample)
    elif arg.task=="validate":
        validate(arg.src_dir, arg.label_dir)
    elif arg.task=="vis":
        vispointclouds(arg.src_dir)
