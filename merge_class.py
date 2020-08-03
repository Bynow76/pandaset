# modify_annotations_txt.py
import glob
import string


CLASS = {'Other_Vehicle_-_Pedicab', 'Construction_Signs', 'Animals_-_Other', 'Motorcycle', 
            'Other_Vehicle_-_Uncommon', 'Other_Vehicle_-_Construction_Vehicle', 'Rolling_Containers', 'Road_Barriers', 
            'Pedestrian', 'Emergency_Vehicle', 'Animals_-_Bird', 'Towed_Object', 'Pedestrian_with_Object', 'Bus', 
            'Semi-truck', 'Bicycle', 'Train', 'Medium-sized_Truck', 'Pylons', 'Car', 'Pickup_Truck', 'Cones', 
            'Motorized_Scooter', 'Tram_/_Subway', 'Signs', 'Personal_Mobility_Device', 'Temporary_Construction_Barriers'}

txt_list = glob.glob('/home/qiuzhongyuan/pointcloud/hesai_new/train/merge_labels/*.txt') # 存储Labels文件夹所有txt文件路径
def show_category(txt_list):
    category_list= []
    for item in txt_list:
        try:
            with open(item) as tdf:
                for each_line in tdf:
                    labeldata = each_line.strip().split(' ') # 去掉前后多余的字符并把其分开
                    category_list.append(labeldata[0]) # 只要第一个字段，即类别
        except IOError as ioerr:
            print('File error:'+str(ioerr))
    print(set(category_list)) # 输出集合

def merge(line):
    each_line=''
    for i in range(len(line)):
        if i!= (len(line)-1):
            each_line=each_line+line[i]+' '
        else:
            each_line=each_line+line[i] # 最后一条字段后面不加空格
    each_line=each_line+'\n'
    return (each_line)

print('before modify categories are:\n')
show_category(txt_list)

for item in txt_list:
    new_txt=[]
    try:
        with open(item, 'r') as r_tdf:
            for each_line in r_tdf:
                labeldata = each_line.strip().split(' ')
                if labeldata[0] in ['Emergency_Vehicle','Bus','Semi-truck', 'Medium-sized_Truck', 'Car','Pickup_Truck',
                                    'Other_Vehicle_-_Pedicab','Other_Vehicle_-_Uncommon','Other_Vehicle_-_Construction_Vehicle']: # 合并汽车类
                    labeldata[0] = labeldata[0].replace(labeldata[0],'TYPE_VEHICLE')
                if labeldata[0] in ['Pedestrian','Pedestrian_with_Object']: # 合并行人类
                    labeldata[0] = labeldata[0].replace(labeldata[0],'TYPE_PEDESTRIAN')
                if labeldata[0] in ['Motorcycle','Bicycle']: # 骑行人
                    labeldata[0] = labeldata[0].replace(labeldata[0],'TYPE_CYCLIST')  
                if labeldata[0] in ['Construction_Signs','Animals_-_Other',  # 忽略类
                                'Rolling_Containers','Temporary_Construction_Barriers',
                                'Road_Barriers','Animals_-_Bird','Towed_Object','Train','Pylons','Cones',
                                'Motorized_Scooter', 'Tram_/_Subway', 'Signs', 'Personal_Mobility_Device']: 
                    continue
                new_txt.append(merge(labeldata)) # 重新写入新的txt文件
        with open(item,'w+') as w_tdf: # w+是打开原文件将内容删除，另写新内容进去
            for temp in new_txt:
                w_tdf.write(temp)
    except IOError as ioerr:
        print('File error:'+str(ioerr))

print('\nafter modify categories are:\n')
