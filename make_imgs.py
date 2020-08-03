# coding: utf-8
from __future__ import division, print_function
from pandaset import DataSet
from pandaset.geometry import *
from PIL import Image
import numpy as np
import os, shutil
import struct
#save_index = 0
save_index = 6560


def extract_data():
    np.random.seed(431972)
    dataset = DataSet('/home/qiuzhongyuan/pointcloud/pandaset')
    sequences = dataset.sequences()
    np.random.shuffle(sequences)
    rate = 0.8
    print(sequences)
    train_sequences = sequences[:int(len(sequences)*rate)]
    val_sequences = sequences[int(len(sequences)*rate):]

    dst_root = '/home/qiuzhongyuan/pointcloud/hesai_imgs'
    assert os.path.exists(dst_root), '%s not exist.' % dst_root

    sequences_map = {'train': train_sequences, 'val': val_sequences}
    global save_index
    labels_name = set()
    '''
    sub_dir = 'train'
    if os.path.exists(os.path.join(dst_root, sub_dir)):
        shutil.rmtree(os.path.join(dst_root, sub_dir))
    os.makedirs(os.path.join(dst_root, sub_dir))
 
    sub_sequences = sequences_map[sub_dir]
    sub_root = os.path.join(dst_root, sub_dir)
    os.makedirs(os.path.join(sub_root, 'pcs'))
    os.makedirs(os.path.join(sub_root, 'labels'))

    for index in sub_sequences:
        seq_data = dataset[index]
        seq_data.lidar.set_sensor(0) #只用360°环视雷达
        seq_data.load_lidar().load_cuboids()
        i=-1
        while True:
            i += 1
            try:
                #pcs_single_frame = seq_data.lidar[i]
                seq_lidars = seq_data.lidar
                pcs_single_frame = seq_data.lidar[i]
            except:
                break
            
            poses = seq_lidars.poses[i]
            
            pcs_single_frame = pcs_single_frame.values
            pcs_single_frame = pcs_single_frame[:,:4]  
            xyz = pcs_single_frame[:,:3]
            intensity = pcs_single_frame[:,3:]
            res = lidar_points_to_ego(xyz, poses)
            res = np.concatenate((res,intensity),axis=-1)
            #temp_x = res[:,0].copy()
            #res[:,0] = res[:,1]
            #res[:,1] = -temp_x
            #res[:,0],res[:,1] = res[:,1],-res[:,0]
            res_new = np.concatenate((res[:,1:2],-res[:,0:1],res[:,2:3],intensity),axis=-1)
            with open(os.path.join(sub_root, 'pcs', "%06d.bin" % save_index), 'wb')as fp:
                for point in res_new:
                    x = struct.pack('f', point[0])
                    y = struct.pack('f', point[1])
                    z = struct.pack('f', point[2])
                    r = struct.pack('f', point[3])
                    fp.write(x)
                    fp.write(y)
                    fp.write(z)
                    fp.write(r)
               
            cuboids_single_frame = seq_data.cuboids[i]
            # 'uuid', 'label', 'yaw', 'stationary', 'camera_used', 'position.x',
            # 'position.y', 'position.z', 'dimensions.x', 'dimensions.y',
            # 'dimensions.z', 'attributes.object_motion', 'cuboids.sibling_id',
            # 'cuboids.sensor_id', 'attributes.rider_status',
            # 'attributes.pedestrian_behavior', 'attributes.pedestrian_age'
            with open(os.path.join(sub_root, 'labels', "%06d.txt" % save_index), 'w') as f:
                cuboids_single_frame = cuboids_single_frame[['label','yaw','position.x','position.y','position.z','dimensions.x','dimensions.y','dimensions.z','cuboids.sensor_id']]
                for inde in cuboids_single_frame.index:
                    label_value = cuboids_single_frame.loc[inde].values[:]
                    sensor_id = int(label_value[-1])
                    if sensor_id>0: #只用360°环视雷达
                        continue
                    cls_name = label_value[0].strip()
                    cls_name = cls_name.replace(' ', '_')
                    labels_name.add(cls_name)

                    
                    box = [label_value[2], label_value[3], label_value[4], label_value[5], label_value[6], label_value[7], label_value[1]]
                   
                    corners = center_box_to_corners(box)                    
                    corners = lidar_points_to_ego(corners, poses)
                    #temp_cx = corners[:,0].copy()
                    #corners[:,0] = corners[:,1]
                    #corners[:,1] = -temp_cx
                    #corners[:,0],corners[:,1] = corners[:,1],-corners[:,0]
                    res_new = np.concatenate((corners[:,1:2],-corners[:,0:1],corners[:,2:3]),axis=-1)
                    x,y,z,l,w,h,yaw = caculate_label(res_new)
                    #save_line = [cls_name, '0', label_value[2], label_value[3], label_value[4], label_value[5], label_value[6], label_value[7], label_value[1]]
                    save_line = [cls_name, '0', x, y, z, l, w, h, yaw]
                    save_line = [str(item) for item in save_line]
                    save_line = ' '.join(save_line) + '\n'
                    f.write(save_line)
               
            save_index += 1
            

        print(index)
        print('total class:', labels_name)
    '''
    sub_dir = 'val'
    #if os.path.exists(os.path.join(dst_root, sub_dir)):
    #    shutil.rmtree(os.path.join(dst_root, sub_dir))
    #os.makedirs(os.path.join(dst_root, sub_dir))

    sub_sequences = sequences_map[sub_dir]
    sub_root = os.path.join(dst_root, sub_dir)
    os.makedirs(os.path.join(sub_root, 'imgs'))
    os.makedirs(os.path.join(sub_root, 'cali'))

    for index in sub_sequences:
        seq_data = dataset[index]        
        seq_data.load()
        i = -1
        while True:
            i += 1
        
            #pcs_single_frame = seq_data.lidar[i]
            #seq_lidars = seq_data.lidar
            #pcs_single_frame = seq_data.lidar[i]
            try:
                front_camera = seq_data.camera['front_camera']
                img0 = front_camera[i]
            except:
                break
            
        
            img0.save(os.path.join(sub_root, 'imgs', "%06d.jpg" % save_index),'JPEG')
            #cv2.imwrite(os.path.join(sub_root, 'imgs', "%06d.jpg" % save_index), img0)

            sl = slice(None, None, 5)  # Equivalent to [::5]
            poses = front_camera.poses
            intrinsics = front_camera.intrinsics
           
            with open(os.path.join(sub_root, 'cali', "%06d.txt" % save_index), 'w') as f:
                

                save_line = [poses[i]['position']['x'],poses[i]['position']['y'],poses[i]['position']['z'], 
                            poses[i]['heading']['w'],poses[i]['heading']['x'],poses[i]['heading']['y'],poses[i]['heading']['z'],
                            intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]
                #save_line = [cls_name, '0', label_value[2], label_value[3], label_value[4], label_value[5],
                #             label_value[6], label_value[7], label_value[1]]
                save_line = [str(item) for item in save_line]
                save_line = ' '.join(save_line) + '\n'
                f.write(save_line)
            save_index += 1
        
        print(index)
        print('total class:', labels_name)
# boundary = {
#     "minX": 0,
#     "maxX": 50,
#     "minY": -25,
#     "maxY": 25,
#     "minZ": -2.73,
#     "maxZ": 1.27
# }
# height = 608
# width = 608
# DISCRETIZATION = (boundary["maxX"] - boundary["minX"])/height#分辨率大小
#
if __name__ == '__main__':
    extract_data()
#     #制作kitti数据集的训练样本标签
#     root = '/home/qiuzhongyuan/pointcloud/second/kitti_1/training'
#     out_dir = './data/kitti'
#     os.makedirs(out_dir, exist_ok=True)
#
#     with open('./data/kitti.names', 'r') as f:
#         lines = f.readlines()
#     lines = [line.strip() for line in lines]
#     type_map = {}
#     for i, line in enumerate(lines):
#         type_map[line] = i
#
#     type_map['Truck'] = type_map['Van'] #此处为了将truck和van合并为一类
#
#     sub_dir = 'train'
#     data_list = os.path.join('./images_sets', sub_dir+'.txt')
#     with open(data_list, 'r') as f:
#         lines = f.readlines()
#     labels = [int(line.strip()) for line in lines]
#
#     dst_txt = os.path.join(out_dir, sub_dir+'.txt')
#     if os.path.exists(dst_txt):
#         os.remove(dst_txt)
#     np.random.shuffle(labels)
#     labels_num = 0
#     for index in labels:
#         label = os.path.join(root, 'label_2', '%06d.txt' % index)
#         calib = Calibration(os.path.join(root, 'calib', '%06d.txt' % index))
#
#         with open(label, 'r') as f:
#             lines = f.readlines()
#         with open(dst_txt, 'a') as f:
#             save_str = 'velodyne/%06d.bin %d %d' % (index, width, height)
#
#             box_num = 0
#             for line in lines:
#                 line = line.strip().split(' ')
#                 cls_ = line[0]
#                 if cls_ not in type_map:
#                     continue
#
#                 cls_index = type_map[cls_]
#                 _,_,_,_,_,_,_,h,w,l,x,y,z,ry = [float(item) for item in line[1:]]
#
#                 x,y,z = camera_to_lidar(x,y,z,calib.V2C, calib.R0, calib.P)
#                 r = -ry - np.pi / 2
#                 if not (boundary['minX'] < x < boundary['maxX'] and boundary['minY'] < y < boundary['maxY']):
#                     continue
#
#                 x,y,l,w = (x-boundary['minX'])/DISCRETIZATION, (y-boundary['minY'])/DISCRETIZATION, l/DISCRETIZATION, w/DISCRETIZATION
#
#                 box_info = [x,y,z,l,w,h,r]
#                 box_info = [str(item) for item in box_info]
#                 box_info = ' ' + str(cls_index) + ' ' + ' '.join(box_info)
#
#                 save_str += box_info
#                 box_num += 1
#
#             if box_num>0:
#                 save_str = str(labels_num) + ' ' + save_str
#                 f.write(save_str + '\n')
#                 labels_num += 1
#
#
#
#     sub_dir = 'val'
#     data_list = os.path.join('./images_sets', sub_dir+'.txt')
#     with open(data_list, 'r') as f:
#         lines = f.readlines()
#     labels = [int(line.strip()) for line in lines]
#
#     dst_txt = os.path.join(out_dir, sub_dir+'.txt')
#     if os.path.exists(dst_txt):
#         os.remove(dst_txt)
#     np.random.shuffle(labels)
#     labels_num = 0
#     for index in labels:
#         label = os.path.join(root, 'label_2', '%06d.txt' % index)
#         calib = Calibration(os.path.join(root, 'calib', '%06d.txt' % index))
#
#         with open(label, 'r') as f:
#             lines = f.readlines()
#         with open(dst_txt, 'a') as f:
#             save_str = 'velodyne/%06d.bin %d %d' % (index, width, height)
#
#             box_num = 0
#             for line in lines:
#                 line = line.strip().split(' ')
#                 cls_ = line[0]
#                 if cls_ not in type_map:
#                     continue
#
#                 cls_index = type_map[cls_]
#                 _,_,_,_,_,_,_,h,w,l,x,y,z,ry = [float(item) for item in line[1:]]
#
#                 x,y,z = camera_to_lidar(x,y,z,calib.V2C, calib.R0, calib.P)
#                 r = -ry - np.pi / 2
#                 if not (boundary['minX'] < x < boundary['maxX'] and boundary['minY'] < y < boundary['maxY']):
#                     continue
#
#                 x,y,l,w = (x-boundary['minX'])/DISCRETIZATION, (y-boundary['minY'])/DISCRETIZATION, l/DISCRETIZATION, w/DISCRETIZATION
#
#                 box_info = [x,y,z,l,w,h,r]
#                 box_info = [str(item) for item in box_info]
#                 box_info = ' ' + str(cls_index) + ' ' + ' '.join(box_info)
#
#                 save_str += box_info
#                 box_num += 1
#
#             if box_num>0:
#                 save_str = str(labels_num) + ' ' + save_str
#                 f.write(save_str + '\n')
#                 labels_num += 1
#
                
                
                
                
                
                
                
                
                
                
                
                
    

