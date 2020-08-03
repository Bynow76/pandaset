# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:38:44 2019

@author: 123456
"""

import numpy as np,cv2
import os, sys, shutil
from skimage import io
import mayavi.mlab as mlab
mlab.options.offscreen = True

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

def draw_gt_boxes3d(gt_boxes3d, colors, fig, line_width=1, draw_text=True, text_scale=(0.5,0.5,0.5)):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,7) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    gt_boxes3d = center_to_corner_box3d(gt_boxes3d, coordinate='lidar') #(n,8,3)
    #print(gt_boxes3d.shape)
    num = len(gt_boxes3d)

    for n in range(num):
        b = gt_boxes3d[n]
        #print("boxes size: ",b.size)
        #print("boxes value: ",b)
        name = 'test'
        color = colors[n]
        #print(n)
        #mlab.text3d(b[4,0], b[4,1], b[4,2], '%s'%name, scale=text_scale, color=color, figure=fig)  #%n
        for k in range(0,4):
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

        center_top = b[4:].mean(0)
        center_front_top = b[4:6].mean(0)
        mlab.plot3d([center_top[0], center_front_top[0]], [center_top[1], center_front_top[1]], [center_top[2], center_front_top[2]], color=(0,1,0), tube_radius=None, line_width=line_width, figure=fig)
    return fig




def draw_lidar(tracking_src, label_dir, dst_dir, index='0004'):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.mkdir(dst_dir)
    os.mkdir(os.path.join(dst_dir, 'viewer1'))
    os.mkdir(os.path.join(dst_dir, 'viewer2'))
    
    class_list = ['Car', 'Van', 'Pedestrian', 'Cyclist']
    COLOR = {'Car':(255,0,0), 'Van':(100,255,100), 'Pedestrian':(255,100,100), 'Cyclist':(130,0, 128)}
    calib_path = os.path.join(tracking_src, 'calib', index+'.txt')
    velodyne_dir = os.path.join(tracking_src, 'velodyne', index)
    
    calib = kitti_util.Calibration(calib_path)
    pcs = os.listdir(velodyne_dir)
    for pc in pcs:
        name = pc[:-4]

        lidar_file = os.path.join(velodyne_dir, pc)
        points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        coords = lidar_to_camera_point(points[:,:3].copy(), V2C=calib.V2C, R0=calib.R0) #(-1, 3)
        coords = np.hstack((coords, np.ones((len(coords), 1)))).T  # (8, 4) -> (4, 8)
        coords = np.matmul(calib.P, coords).T

        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]
        mask = (coords[:, 0]>=0) & (coords[:, 0]<1242) & (coords[:, 1]>=0) & (coords[:, 1]<375)
        points = points[mask]
        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1242, 500))
        color = points[:, 3]
        color = np.clip(color+0.3, 0, 1)
        mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color, color=None, mode='point', colormap='GnBu', scale_factor=1, figure=fig)
        
        labels = kitti_util.read_label(os.path.join(label_dir, name+'.txt'))
        for obj in labels:
            if obj.type not in class_list:
                continue
            x, y, z = obj.t #carema coordinate
            center = np.array([[x,y,z]], dtype=np.float32)
            center = np.hstack((center, np.ones((len(center), 1)))).T  # (1, 4) -> (4, 1)
            center = np.matmul(calib.P, center).flatten()
    
            center[0] /= center[2]
            center[1] /= center[2]
            mask = (center[0]>=0) & (center[0]<1242) & (center[1]>=0) & (center[1]<375)
            if not mask:
                continue
            rz = -obj.ry - np.pi / 2
            x, y, z = camera_to_lidar(x, y, z, V2C=calib.V2C,R0=calib.R0,P2=calib.P)#lidar coordinate
            boxes_center = np.array([x,y,z, obj.h, obj.w, obj.l, rz], dtype=np.float32).reshape(1, -1)
            color = tuple([float(v)/256 for v in COLOR[obj.type]])
            fig = draw_gt_boxes3d([obj.type], boxes_center, [color], fig, line_width=1.5, draw_text=True, text_scale=(0.5,0.5,0.5))
    
            
        mlab.view(azimuth=130, elevation=45, focalpoint=[30, 10, 0], distance=60.0, figure=fig)
        mlab.show()
        #mlab.savefig(os.path.join(dst_dir, 'viewer1', name+'.jpg'), figure=mlab.gcf())
        
        #elevation:俯仰角，向下0°，向上180°
        #mlab.view(azimuth=180, elevation=45, focalpoint=[25, 0, 0], distance=70.0, figure=fig)
        #mlab.show()
        #mlab.savefig(os.path.join(dst_dir, 'viewer2', name+'.jpg'), figure=mlab.gcf())
        #mlab.close(fig)

if __name__ == '__main__':
    
    points = np.fromfile('/Users/pengfeima/Desktop/Pandaset/003000.bin', dtype=np.float32).reshape(-1, 4)
    #points = points[points[:,2]>0]
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1300, 600))
    color = points[:, 3]
    color = np.clip(color+0.3, 0, 1)
    mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color, color=None, mode='point', colormap='GnBu', scale_factor=1, figure=fig)
    
    with open('/Users/pengfeima/Desktop/Pandaset/003000.txt', 'r') as f:
        lines = f.readlines()
        
    print(len(lines))
    
    #COLOR = {'TYPE_VEHICLE':(1.0,0,0), 'TYPE_PEDESTRIAN':(0.0,0.0,1.0), 'TYPE_SIGN':(1.0,0.35,0.35), 'TYPE_CYCLIST':(0.55,0, 0.5)}
    COLOR = {'Unknown':(0,0,0),'Vehicle':(1.0,0,0),'Pedestrian':(0.0,0.0,1.0),'Bicycle':(0.55,0, 0.5)}
    i = 0
    labels = []
    for line in lines:
        line = line.strip().split(' ')
        cls_type = line[0]
        x = float(line[2])
        y = float(line[3])
        z = float(line[4])
        l = float(line[6])
        w = float(line[5])
        h = float(line[7])
        r = float(line[8])

        if np.abs(x)<=80 and np.abs(y)<=50:

            boxes_center = np.array([x,y,z, h, w, l, r], dtype=np.float32).reshape(1, -1)
            #color = COLOR[cls_type]
            color = (0.0,0.0,1.0)
            fig = draw_gt_boxes3d(boxes_center, [color], fig, line_width=1.5, draw_text=True, text_scale=(0.5,0.5,0.5))
        i += 1
        index = '{}/{}'.format(i,len(lines))
        print(index)
        #break

    mlab.view(azimuth=180, elevation=45, focalpoint=[0, 0, 0], distance=120.0, figure=fig)
    mlab.show()
    mlab.savefig('./Pandaset/003000.jpg', figure=mlab.gcf())
    mlab.close(fig)
    
'''


if __name__ == '__main__':
    test_bin_path='/Users/pengfeima/Downloads/Apollo_data/tracking_test_pcd_1/result_9048_2_frame/'
    label_path='./pred_labels/'
    res_img_path='./pc_img/'
    #test_bin_path='/workspace/CNNSeg-master/data/'
    filenames = os.listdir(test_bin_path)
    for filename in filenames:
        points = np.fromfile(test_bin_path+filename, dtype=np.float32).reshape(-1, 4)    
        #points = points[points[:,2]>0]
        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1280, 720))
        color = points[:, 3]
        color = np.clip(color+0.3, 0, 1)
        mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color, color=None, mode='point', colormap='GnBu', scale_factor=1, figure=fig)
        
        with open(label_path+filename[:-4]+'.txt', 'r') as f:
            lines = f.readlines()
            
        #print(len(lines))
        
        #COLOR = {'TYPE_VEHICLE':(1.0,0,0), 'TYPE_PEDESTRIAN':(0.0,0.0,1.0), 'TYPE_SIGN':(1.0,0.35,0.35), 'TYPE_CYCLIST':(0.55,0, 0.5)}
        #COLOR = {'0':(0,0,0),'1':(1.0,0,0),'2':(1.0,0,0),'3':(0.0,0.0,1.0),'4':(0.55,0, 0.5),'5':(1.0,0.35,0.35),'6':(1.0,0.35,0.35)}
        COLOR = {'Unknown':(0,0,0),'Vehicle':(1.0,0,0),'Pedestrian':(0.0,0.0,1.0),'Bicycle':(0.55,0, 0.5)}
        # For object_type, 1 for small vehicles, 2 for big vehicles, 3 for pedestrian, 
        #   4 for motorcyclist and bicyclist, 5 for traffic cones and 6 for others
        i = 0
        labels = []
        for line in lines:
            line = line.strip().split(' ')
            cls_type = line[0]
            x = float(line[1])
            y = float(line[2])
            z = float(line[3])
            l = float(line[4])
            w = float(line[5])
            h = float(line[6])
            r = float(line[7])

            boxes_center = np.array([x,y,z, h, w, l, r], dtype=np.float32).reshape(1, -1)
            color = COLOR[cls_type]
            fig = draw_gt_boxes3d(boxes_center, [color], fig, line_width=1.5, draw_text=True, text_scale=(0.5,0.5,0.5))
            i += 1


        mlab.view(azimuth=180, elevation=45, focalpoint=[0, 0, 0], distance=120.0, figure=fig)
        
        # mlab.show()
        mlab.savefig(res_img_path+filename[:-4]+'.jpg', figure=mlab.gcf())
        mlab.close(fig)
'''       

    