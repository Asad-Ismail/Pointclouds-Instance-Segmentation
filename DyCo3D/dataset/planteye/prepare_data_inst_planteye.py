'''
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
'''

import glob, plyfile, numpy as np, multiprocessing as mp, torch, json, argparse
#import open3d as o3d



# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1,2], start=1):
    remapper[x] = i

parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train / val / test)', default='train')
parser.add_argument('--src_dir', help='data directory contianing ply and labels', default='/media/asad/adas_cv_2/pointclouds-dev/DyCo3D/dataset/planteye/train')
opt = parser.parse_args()

split = opt.data_split
print('data split: {}'.format(split))
files = sorted(glob.glob(opt.src_dir + '/*.ply'))
#/media/asad/adas_cv_2/pointclouds-dev/DyCo3D/dataset/planteye/train


def f_test(fn):
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    torch.save((coords, colors), fn[:-4]+ '_inst_nostuff.pth')
    print('Saving to ' + fn[:-15] + '_inst_nostuff.pth')


def f(fn):

    label_file = fn[:-4] + '.txt'
    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1
    load_labels=np.loadtxt(label_file,dtype=np.compat.long)
    ins_label=load_labels%1000
    #print(np.unique(ins_label))
    sem_labels=load_labels//1000
    #sem_labels=sem_labels.astype(np.float64)
    #sem_labels = remapper[sem_labels]
    #print(np.unique(sem_labels))
    ins_label[np.where(ins_label==0)]=-100
    #print(np.unique(ins_label))
    #instance_labels = np.ones(sem_labels.shape[0]) * -100
    torch.save((coords, colors, sem_labels, ins_label),fn[:-4]+'_inst_nostuff.pth')
    print('Saving to ' + fn[:-4]+'_inst_nostuff.pth')

#for fn in files:
#    f(fn)

p = mp.Pool(processes=mp.cpu_count())
if opt.data_split == 'test':
    p.map(f_test, files)
else:
    p.map(f, files)
p.close()
p.join()
