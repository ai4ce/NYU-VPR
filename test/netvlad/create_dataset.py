import utm
import os
import scipy.io as sio 
from collections import namedtuple
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
path_1 = './netvlad/front_test_images_paths.txt'
path_2 = './washington-square/washington-square_all.csv'
path_3 = './netvlad/front_train_images_paths.txt'
posDistThr = 25
posDistSqThr = 625
nonTrivPosDistSqThr = 100
utmDb = []
utmQ = []
db1Image = []
q1Image = []
db2Image = []
q2Image = []
numDb = 0
numQ = 0
whichSet = path_1.split('/')[3].split('_')[0]
dataset = 'washingtonSquare'
f_1 = open(path_1, 'r', encoding='utf-16')
f_1_name = f_1.readlines()
f_2 = open(path_2)
f_name = f_2.readlines()    
f_3 = open(path_3, 'r', encoding='utf-16')
f_3_name = f_3.readlines()   
for item in range(0, len(f_1_name)):        
    name_temp = f_1_name[item].split('/')[4].split('.jpg')[0]+'.jpg'
    if os.path.isfile('./data_1/washington-square/front1/' + name_temp):       
#        name_temp_0 = f_1_name[item].split('/')[4].split('.')[0]+'.jpg'
        q1Image.append('./data_1/washington-square/front1/' + name_temp)        
        q2Image.append('./data_1/washington-square/front2/' + name_temp)
        for row in range(1, len(f_name)):
 #           name_image = f_name[row].split(',')[14].split('/')[5].split('"')[0]     
            name_image = f_name[row].split(',')[15].split('/')[5].split('"')[0] 
            if name_image == name_temp:
#                lat = f_name[row].split(',')[15]
#                lon = f_name[row].split(',')[16]
                lat = f_name[row].split(',')[16]
                lon = f_name[row].split(',')[17]
                gps = (float(lat), float(lon))
                utmQ.append(utm.from_latlon(gps[0], gps[1])[0:2])
                numQ += 1
                print(numQ)
                break
    else:
                print(name_temp)
for item in range(0, len(f_3_name)):        
    name_temp = f_3_name[item].split('/')[4].split('.jpg')[0]+'.jpg'
    if os.path.exists('/home/jianzhelin/data_1/washington-square/front1/' + name_temp):      
#        name_temp_0 = f_3_name[item].split('/')[5].split('.')[0]+'.jpg'
        db1Image.append('./data_1/washington-square/front1/' + name_temp) 
        db2Image.append('./data_1/washington-square/front2/' + name_temp)         
        for row in range(1, len(f_name)):
 #           name_image = f_name[row].split(',')[14].split('/')[5].split('"')[0]     
            name_image = f_name[row].split(',')[15].split('/')[5].split('"')[0] 
            if name_image == name_temp:
#                lat = f_name[row].split(',')[15]
#                lon = f_name[row].split(',')[16]
                lat = f_name[row].split(',')[16]
                lon = f_name[row].split(',')[17]
                gps = (float(lat), float(lon))
                utmDb.append(utm.from_latlon(gps[0], gps[1])[0:2])
                numDb += 1
                print(numDb)
                break
    else:
                print(name_temp)
sio.savemat('./front1_test_data.mat',{'whichSet': whichSet, 'dataset':dataset,'dbImage':db1Image,   
            'qImage':q1Image, 'utmDb':utmDb, 'numDb':numDb, 'utmQ':utmQ, 'numQ':numQ, 'posDistThr':posDistThr, 
            'posDistSqThr':posDistSqThr, 'nonTrivPosDistSqThr':nonTrivPosDistSqThr})
sio.savemat('./front2_test_data.mat',{'whichSet': whichSet, 'dataset':dataset,'dbImage':db2Image,   
            'qImage':q2Image, 'utmDb':utmDb, 'numDb':numDb, 'utmQ':utmQ, 'numQ':numQ, 'posDistThr':posDistThr, 
            'posDistSqThr':posDistSqThr, 'nonTrivPosDistSqThr':nonTrivPosDistSqThr})
