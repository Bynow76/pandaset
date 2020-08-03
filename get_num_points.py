# -*- coding: utf-8 -*-
import os, sys
import math
import numpy as np
from scipy.spatial import Delaunay
from skimage import io

def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return box3d_roi_inds.sum()
    
def center_to_corner_box3d(boxes_center):
	# (N, 7) -> (N, 8, 3) #中心点坐标转换为立方体8个角坐标

	N = boxes_center.shape[0]
	ret = np.zeros((N, 8, 3), dtype=np.float32)

	for i in range(N):
		box = boxes_center[i]
		translation = box[0:3]#中心坐标
		size = box[3:6]  #框的尺寸
		rotation = [0, 0, box[-1]]

		h, w, l = size[0], size[1], size[2]
		trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
			[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
			[w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])
			#[0, 0, 0, 0, h, h, h, h]])

		# re-create 3D bounding box in velodyne coordinate system
		yaw = rotation[2]
		rotMat = np.array([
			[np.cos(yaw), -np.sin(yaw), 0.0],
			[np.sin(yaw), np.cos(yaw), 0.0],
			[0.0, 0.0, 1.0]])
		cornerPosInVelo = np.dot(rotMat, trackletBox) + \
			np.tile(translation, (8, 1)).T
		box3d = cornerPosInVelo.transpose()
		ret[i] = box3d

	return ret
    
    
src_path = "/home/qiuzhongyuan/pointcloud/hesai_new/val/"
dst_path = os.path.join(src_path, 'labels_1')
if not os.path.exists(dst_path):
    os.mkdir(dst_path)


pcs_dir = os.path.join(src_path, 'pcs')
labels_dir = os.path.join(src_path, 'merge_labels')
pcs = os.listdir(pcs_dir)
pcs = sorted(pcs)

for pc in pcs:
    name = pc[:-4]
    pc_data = np.fromfile(os.path.join(pcs_dir, pc), dtype=np.float32).reshape(-1, 4)
    #pc_data = io.imread(os.path.join(pcs_dir, pc))
    '''
    mask = pc_data[:,:,4]>0.5
    x = pc_data[:,:,0][mask]
    y = pc_data[:,:,1][mask]
    z = pc_data[:,:,2][mask]
    intensity = pc_data[:,:,3][mask]
    pc_data = np.stack([x,y,z,intensity]).reshape(4,-1).T
    '''
    
    with open(os.path.join(labels_dir, name+'.txt'), 'r') as f:
        lines = f.readlines()
    
    results = []
    for line in lines:
        line = line.strip().split(' ')
        x = float(line[2])
        y = float(line[3])
        z = float(line[4])
        l = float(line[5])
        w = float(line[6])
        h = float(line[7])
        r = float(line[8])
        boxes_center = np.array([x,y,z, h, w, l, r], dtype=np.float32).reshape(1, -1)
        gt_boxes3d = center_to_corner_box3d(boxes_center) #(n,8,3)
        gt_boxes3d = gt_boxes3d[0]
        num_points = extract_pc_in_box3d(pc_data, gt_boxes3d)
        line.append(str(num_points))
        results.append(line)
    
    with open(os.path.join(dst_path, name+'.txt'), 'w') as f:
        for line in results:
            line = ' '.join(line) + '\n'
            f.write(line)
    print(name, 'is ok!')

















