
## [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Asad-Ismail/Pointclouds-Instance-Segmentation/issues)[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FAsad-Ismail%2FPointclouds-Instance-Segmentation&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# Pointclouds Annotation and Instance Segmentation 
End to End pipeline for Annotation and Instance Segmentation of point clouds

## Motivation
In this work we will build End to End Pipeline for 3D pointclouds instance segmentation of plants for plant phenotyping. Plant seperation/Instance segmentation is the first step for high throughput phenotyping of plants. Traditional phenotyping requires plants to be removed/seperated mannually and then perform phenotyping. This process limits the speed of phenotyping and we can perform much faster phenotyping if we can seperate the plants using software. Fortunately, with recent advancements in deep learning based instance segmentation [1,3] we can perform 3D instance segmentation with high accuracy which was not possible using traiditional methods of instance semgmentation like DBSCAN or Graph based clustering.  Thre are two major tasks we will address in this work\
\
**1) Build a pipeline for obtaining 3D point cloud annotated dataset for instance segmentation:** \
    3D point clouds take much more time to be segmented and hence the cost of their annotation can be 10x compare to 2D images. As a result there are not so many                   publically available datasets for point cloud instance segmentation.

**2) Benchmark and compare state-of-the-art 3D point clouds instance segmentation networks:** \
    Point clouds provide interesting challenge for instance segmentation becuase of their unstuctured, unordered and sparse nature which requires special properties from neural networks like permuation invariance, transformation invariance and point interactions. Unlike images there are not enough  point clouds dataset publically available from different domains. Most of the point cloud datasets are from autonomous driving dmian and some from indoor point clouds domain (containing furniture walls e.t.c) for instance segmentation. It is not clear how deep neural networks trained and benchmarked on these datasets will perform on     out of domain datasets. In this work we annotate 200 scans of 3D point clouds of plants to perform plant seperation and evaluate different state of the art algortihhms on this custom dataset 

## Pipeline Summary
1) First the point clouds are preprocessed to downsamaple from millions of point clouds to few hundered thousands. We use voxel size of 3 cm to downsample the pointclouds \
Below we have an example, on left we have original point clouds with 3.9 million points on right we have preprocessed point clouds with removed tray (based on height threshold) and downsampled point clouds resulting in 134000 points 

  <p align="center">
    <img src="images/plants_preprocess.gif" alt="pruning" />
  </p>
   <p align="center"> 
    
2) Train the deep neural network(PointGroup, Dyco3D) and perform hyper parameter search (Bayseian based) with the preprocessed data 
    
3) Use the trained network to make inference on new data. Example Input Point cloud, Labels and Prediction is shown below
    
  <p align="center">
    <img src="images/image.png" alt="pruning" />
  </p>
   <p align="center">


### Stucture

    .
    ├── annotation              # 3D point cloud annotation using sagemaker 
    ├── algorithms              # Algorithms for 3D instance segmentation               
    ├── dataset                 # Dataset for training  and validation
    ├── utils                   # General useful scripts for viusualization and pre/post processing point clouds                     
    └── ...

See the correspoding direcory for more detail for each.

### Requirements

python>=3.7\
open3d\
In addition please see requirements of each algorithm to see the requiremnt of each algorithm.

    
### Point Cloud Annotation:
See annotation directory for details on Amazon sagemaker pipeline for 3D point cloud annotation. 


**For questions and if something is not working please open an issue or send a pull request**
  
### References
```
1) @article{jiang2020pointgroup,
  title={PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation},
  author={Jiang, Li and Zhao, Hengshuang and Shi, Shaoshuai and Liu, Shu and Fu, Chi-Wing and Jia, Jiaya},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}}
2) @inproceedings{He2021dyco3d,
  title     =   {{DyCo3d}: Robust Instance Segmentation of 3D Point Clouds through Dynamic Convolution},
  author    =   {Tong He and Chunhua Shen and Anton van den Hengel},
  booktitle =   {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      =   {2021}
}
3) @inproceedings{liang2021instance,
  title={Instance Segmentation in 3D Scenes using Semantic Superpoint Tree Networks},
  author={Liang, Zhihao and Li, Zhihao and Xu, Songcen and Tan, Mingkui and Jia, Kui},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2783--2792},
  year={2021}
}
  


