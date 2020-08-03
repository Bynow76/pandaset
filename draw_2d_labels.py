from __future__ import division, print_function
from pandaset import DataSet
from pandaset.geometry import *
from PIL import Image
import numpy as np
import os, shutil
import struct
import transforms3d as t3d
import cv2
from skimage import io
from math import *
import numpy as np
import time,math
import os
import json

def center_to_corner_box3d(boxes_center, coordinate='lidar'):
	# (N, 7) -> (N, 8, 3) #中心点坐标转换为立方体8个角坐标
	if coordinate != 'lidar':
		raise RuntimeError('error')
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
def draw_projected_box3d(image, qs, color=(255,0,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    #qs = qs.astype(np.int32)
    for k in range(0,4):
       # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i,j=k,(k+1)%4
       # use LINE_AA for opencv3
        p1, p2 = (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1])
        cv2.line(image, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), color, thickness)


        i,j=k+4,(k+1)%4 + 4
        p1, p2 = (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1])
        cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)

        i,j=k,k+4
        p1, p2 = (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1])
        cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)

    return image

def _heading_position_to_mat(heading, position):
    quat = np.array([heading["w"], heading["x"], heading["y"], heading["z"]])
    pos = np.array([position["x"], position["y"], position["z"]])
    transform_matrix = t3d.affines.compose(np.array(pos),
                                           t3d.quaternions.quat2mat(quat),
                                           [1.0, 1.0, 1.0])
    return transform_matrix
def projection_lidar_img(lidar_points, camera_data, camera_pose, camera_intrinsics, filter_outliers=False):
    camera_heading = camera_pose['heading']
    camera_position = camera_pose['position']
    camera_pose_mat = _heading_position_to_mat(camera_heading, camera_position)
     
    trans_lidar_to_camera = np.linalg.inv(camera_pose_mat)
    points3d_lidar = lidar_points
    points3d_camera = trans_lidar_to_camera[:3, :3] @ (points3d_lidar.T) + \
                        trans_lidar_to_camera[:3, 3].reshape(3, 1)
    #print("points3d_camera: ",points3d_camera.shape, points3d_camera)
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = camera_intrinsics[0]
    K[1, 1] = camera_intrinsics[1]
    K[0, 2] = camera_intrinsics[2]
    K[1, 2] = camera_intrinsics[3]
    #print(K)
    inliner_indices_arr = np.arange(points3d_camera.shape[1])
    if filter_outliers:
        condition = points3d_camera[2, :] > 0.0
        points3d_camera = points3d_camera[:, condition]
        inliner_indices_arr = inliner_indices_arr[condition]

    points2d_camera = K @ points3d_camera
    #print("points2d_camera: ",points2d_camera.shape, points2d_camera)
    #points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T
    
    if filter_outliers:
        image_h, image_w, channel = camera_data.shape
        print(image_w,image_h,channel)
        condition = np.logical_and(
            (points2d_camera[:, 1] < image_h) & (points2d_camera[:, 1] > 0),
            (points2d_camera[:, 0] < image_w) & (points2d_camera[:, 0] > 0))
        points2d_camera = points2d_camera[condition]
        points3d_camera = (points3d_camera.T)[condition]
        inliner_indices_arr = inliner_indices_arr[condition]
    return points2d_camera, points3d_camera, inliner_indices_arr

if __name__ == '__main__':
    with open('/home/qiuzhongyuan/pointcloud/hesai/val/labels_num_gt5/008239.txt', 'r') as f: # predict labels
        lines = f.readlines()
    camera_data = cv2.imread('/home/qiuzhongyuan/pointcloud/hesai_imgs/val/imgs/008239.jpg')   # imgs
    with open('/home/qiuzhongyuan/pointcloud/hesai_imgs/val/cali/008239.txt','r') as fp:  # calib
        intrics = fp.readline().split(' ')
    with open('/home/qiuzhongyuan/pointcloud/hesai_imgs/val/lidar_pose/008239.txt','r') as fpp:
        lidar =   fpp.readline().split(' ')
    camera_pose = {'heading':{'w':float(intrics[3]),'x':float(intrics[4]),'y':float(intrics[5]),'z':float(intrics[6])}, 
                    'position':{'x':float(intrics[0]),'y':float(intrics[1]),'z':float(intrics[2])}}
    camera_intrinsics = [float(intrics[7]),float(intrics[8]),float(intrics[9]),float(intrics[10])]
    lidar_pose = {'heading':{'w':float(lidar[3]),'x':float(lidar[4]),'y':float(lidar[5]),'z':float(lidar[6])}, 
                    'position':{'x':float(lidar[0]),'y':float(lidar[1]),'z':float(lidar[2])}}
    lidar_pose_mat = _heading_position_to_mat(lidar_pose['heading'], lidar_pose['position'])
    transform_matrix = np.linalg.inv(lidar_pose_mat)
    #lidar_pose_lina = np.linalg.inv(transform_matrix[:3,:3])

    i = 0
    for line in lines:
        line = line.strip().split(' ')
        cls_type = line[0]
        x = float(line[2])
        y = float(line[3])
        z = float(line[4])
        l = float(line[5])
        w = float(line[6])
        h = float(line[7])
        r = float(line[8])
        boxes_center = np.array([x,y,z, h, w, l, r], dtype=np.float32).reshape(1, -1)
        
        lidar_points = center_to_corner_box3d(boxes_center).reshape(8,-1) 
                    
        #print('lidar_points: ',lidar_points)
        for item in lidar_points:
            item[0], item[1] = -item[1], item[0]
        ego_points = (lidar_points.T - transform_matrix[:3, [3]])
        world_points = (lidar_pose_mat[:3,:3] @ ego_points).T
        #world_points = (lidar_pose_lina @ ego_points).T
        #print('world_points: ',world_points)
        points2d_camera, points3d_camera, inliner_indices_arr = projection_lidar_img(world_points, 
                        camera_data, camera_pose,camera_intrinsics)
        print('points2d_camera: ', points2d_camera.shape, points2d_camera)
        #print('points3d_camera: ', points3d_camera.shape, points3d_camera)
        #print('inliner_indices_arr: ', inliner_indices_arr.shape, inliner_indices_arr)    
        if np.min(points2d_camera[2, :] >=0 ):
            points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T
        else:
            i += 1
            continue       
        
        draw_projected_box3d(camera_data, points2d_camera)
        #if i > 3:
        #    break
        i += 1
        index = '{}/{}'.format(i,len(lines))
        print(index)
    cv2.imwrite('/home/qiuzhongyuan/pointcloud/hesai_imgs/a_008239.jpg', camera_data)