import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import BoxMode


# Detectron2 config put in a seperate module
class detectroninference:
    def __init__(self,model_path,num_cls=1,name_classes=["pepp"]):
        self.cfg = get_cfg()
        self.cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml")
        self.cfg.DATASETS.TRAIN = ("cuc_train",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 10
        self.cfg.MODEL.WEIGHTS = model_path  # Let training initialize from model zoo
        self.cfg["MODEL"]["ANCHOR_GENERATOR"]["ASPECT_RATIOS"][0]=[0.5,1.0,1.5]
        #cfg["INPUT"]["MASK_FORMAT"]='bitmask'
        self.cfg["INPUT"]["RANDOM_FLIP"]="horizontal"
        #cfg["INPUT"]["RANDOM_FLIP"]="horizontal"
        self.cfg["INPUT"]["ROTATE"]=[-2.0,2.0]
        self.cfg["INPUT"]["LIGHT_SCALE"]=2
        self.cfg["INPUT"]["Brightness_SCALE"]=[0.5,1.5]
        self.cfg["INPUT"]["Contrast_SCALE"]=[0.5,2]
        self.cfg["INPUT"]["Saturation_SCALE"]=[0.5,2]
        #cfg["BASE_LR"]=1e-5
        self.cfg.SOLVER.IMS_PER_BATCH = 1
        self.cfg.SOLVER.BASE_LR = 1e-3  # pick a good LR
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 500
        self.cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
        self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES=1
        self.cfg.MODEL.RETINANET.NUM_CLASSES=1
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95
        self.predictor = DefaultPredictor(self.cfg)
        self.cuc_metadata = MetadataCatalog.get("pepp").set(thing_classes=name_classes)


    
    def apply_mask(self,mask,img):
        all_masks=np.zeros(mask.shape,dtype=np.uint8)
        all_patches=np.zeros((*mask.shape,3),dtype=np.uint8)
        """Apply the given mask to the image."""
        for i in range(all_masks.shape[0]):
                all_masks[i][:, :] = np.where(mask[i] == True,255,0)
                for j in range(3):
                    all_patches[i][:, :,j] = np.where(mask[i] == True,img[:,:,j],0)
        return all_masks,all_patches


    def pred(self,img):
        orig_img=img.copy()
        height,width=img.shape[:2]
        outputs = self.predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        #v = Visualizer(img[:, :, ::-1],
        #                metadata=self.cuc_metadata, 
        #                scale=0.3, 
        #                instance_mode=ColorMode.IMAGE_BW)   # remove the colors of unsegmented pixels. This option is only available for segmentation models
           
        #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        masks = np.asarray(outputs["instances"].to("cpu").pred_masks)
        #masks,patches=self.apply_mask(masks,orig_img)
        classes=outputs["instances"].pred_classes.to("cpu").numpy()
        boxes=(outputs["instances"].pred_boxes.to("cpu").tensor.numpy())
        #print(c)
        return masks,boxes,classes
        #return out.get_image()[:, :, ::-1],masks,patches,boxes,classes,outputs["instances"].scores.to("cpu").numpy()

def get_labels(filename):
    with open(filename,"r") as f:
        all_lines=f.readlines()
    i=0
    while(i<len(all_lines)):
        if(i>len(all_lines)):
            break
        line=all_lines[i].split(",")
        label=line[0]
        file=line[3]
        first_point=None
        second_point=None
        #print(i)
        #print(line)
        if label=="head":
            first_point=(int(line[1]),int(line[2]))
        elif label =="tail":
            second_point=(int(line[1]),int(line[2]))
        i+=1
        if(i<len(all_lines)):
            line2=all_lines[i].split(",")
            if line2[3]==file:
                if line2[0]=="head":
                    first_point=(int(line2[1]),int(line2[2]))
                elif line2[0] =="tail":
                    second_point=(int(line2[1]),int(line2[2]))
                i+=1
                #print(line2)
        #print(first_point,second_point)
        points[file]=[first_point,second_point]
             



def get_y(fname):
        f=fname.name
        ps=points[f]
        im = PILImage.create(fname)
        out=[]
        # put out of bound in bound
        for x in ps:
            if x is None:
                x=(-1,-1)
            if x[0] > im.size[0]:
                x[0]=im.size[0]-3
            if x[1] > im.size[1] :
                x[1]=im.size[1]-3
            out.append(x)
        out=np.array(out,dtype=np.float)
        return tensor(out)

def fast_AILearner(model):
    learn_inf = load_learner(model)
    return learn_inf
    item_tfms = [Resize(224, method='squish',pad_mode="zeros")]
    #batch_tfms = [Flip(), Rotate(), Zoom(), Warp(),ClampBatch()]
    batch_tfms = [Flip(pad_mode="zeros"), Rotate(pad_mode="zeros"), Warp(pad_mode="zeros")]
    dblock = DataBlock(blocks=(ImageBlock, PointBlock),
                   get_items=get_image_files,
                   splitter=RandomSplitter(),
                   get_y=get_y,
                   item_tfms=item_tfms,
                   batch_tfms=batch_tfms)
    bs=4
    print(img_dir)
    dls = dblock.dataloaders(img_dir, bs=bs)
    dls.c = dls.train.after_item.c
    learn = cnn_learner(dls, models.resnet34, lin_ftrs=[100], ps=0.01,concat_pool=True,loss_func=MSELossFlat())
    learn.load(model)
    return learn

def scale_predictions(pred,imgShape,pred_size=224):
    scale_x=imgShape[1]/pred_size
    scale_y=imgShape[0]/pred_size
    pred=pred[0]
    pred[:,0]=pred[:,0]*scale_x
    pred[:,1]=pred[:,1]*scale_y
    return pred


def get_angle(p0, p1, p2):
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def check_points(cropedPatches,points):
    fig = plt.figure(figsize=(10, 10))
    plt.title("Points ResNet34",loc='center')
    #plt.rcParams['figure.figsize'] = [45, 45]
    nrows=int(np.ceil(len(points.keys())/2))
    ncols=2
    
    for i,img in enumerate(cropedPatches):
        img=img.copy()
        pointsize=img.shape[0]//20
        assert img is not None, "Image not found!!"
        if points[i][0] is not None:
            cv2.circle(img,points[i][0],pointsize,(255,255,0),-1)
        if points[i][1] is not None:
            cv2.circle(img,points[i][1],pointsize,(255,0,255),-1)
        if points[i][0] is not None and points[i][1] is not None:
            #cv2.line(img,tuple(points[i][0]),tuple(points[i][1]),(255,255,255),2)
            ref=(img.shape[1],points[i][1][1])
            #cv2.line(img,tuple(points[i][1]),ref,(255,255,255),2)
        ax = fig.add_subplot(nrows,ncols,i+1)
        ax.imshow(img[...,::-1])
        #plt.tight_layout()
    return fig

def apply_rotation(cropedPatches,predPoints,th=5):
    corrected_imgs=[]
    for i,crop in enumerate(cropedPatches):
        img=crop.copy()
        h,w=img.shape[0:2]
        head,tail=None,None
        if predPoints[i][0] is not None and predPoints[i][0][0]>0 and predPoints[i][0][1]>0:
            head=list(predPoints[i][0])
        if predPoints[i][1] is not None and predPoints[i][1][0]>0 and predPoints[i][1][1]>0:
            tail=list(predPoints[i][1])
        #print(head,tail)
        if head is not None and tail is not None:
            #print(i)
            if abs(head[0]-w)<th:
                head[0]-=th
                #print(f"Lower threshold for head index {i}")
            if abs(tail[0]-w)<th:
                tail[0]-=th
            ref=(img.shape[1],tail[1]) 
            #cv2.circle(img,tuple(head),5,(255,0,0),-1)
            #cv2.circle(img,tuple(tail),5,(0,255,0),-1)
            #cv2.line(img,tuple(head),tuple(tail),(255,255,255),2)
            #cv2.line(img,tail,ref,(255,255,255),2)
            angle=get_angle(head,tail,ref)
            #rotated=rotate_image(vis_img,-angle2,tail)
            rotated=rotate_image(img,-angle)
            corrected_imgs.append(rotated)
    return corrected_imgs
