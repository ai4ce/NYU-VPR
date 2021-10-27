import cv2
import math
import numpy as np
from tqdm import tqdm
import csv


def similarity_score(image1, image2):
    feature_detector = cv2.xfeatures2d.SIFT_create()
    
    kp1, des1 = feature_detector.detectAndCompute(image1,None)
    kp2, des2 = feature_detector.detectAndCompute(image2,None)
    matches = None
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.9 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC)

    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    if len(pts1) > 20:
        return len(pts2)
    return 0


def mix(train, test, index):
    gps_imgs = gps(test)
    # print(img3)
    img3 = []
    img4 = []
    for i in range(4):
        temp = cv2.imread(gps_imgs[i])
        temp = cv2.resize(temp, (320, 240))
        img3.append(temp)
    for i in range(4, 8):
        temp = cv2.imread(gps_imgs[i])
        temp = cv2.resize(temp, (320, 240))
        img4.append(temp)
    img1 = cv2.imread(test)
    img2 = cv2.imread(train)

    # img1 = cv2.resize(img1, (640, 480))

    base_score = similarity_score(img1, img1)
    num = similarity_score(img1, img2)
    # gps_num = similarity_score(img1, img3)
    score = math.ceil(num / float(base_score) * 1000.0) / 1000.0
    # gps_score = math.ceil(gps_num / float(base_score) * 1000.0) / 1000.0

    tint_mask = np.ones((480, 640, 3))
    tint_mask[:,:,0] = 0.7
    tint_mask[:,:,1] = 0.7

    if score < 0.03:
        img2 = img2 * tint_mask
    # if gps_score < 0.03:
    #     img3 = img3 * tint_mask

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255,255,255)
    thickness = -1

    cv2.rectangle(img1, (0, 0), (170, 40), color, thickness)
    cv2.rectangle(img2, (0, 0), (170, 40), color, thickness)
    cv2.rectangle(img2, (450, 0), (640, 40), color, thickness)
    # cv2.rectangle(img3, (0, 0), (170, 40), color, thickness)
    # cv2.rectangle(img3, (450, 0), (640, 40), color, thickness)

    cv2.putText(img1, 'Query Image', (5, 30), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img2, 'Train Image', (5, 30), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    # cv2.putText(img3, 'GPS Image', (5, 30), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(img2, 'Score: '+str(score), (455, 30), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    # cv2.putText(img3, 'Score: '+str(gps_score), (455, 30), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    disp1 = np.concatenate((img3[0], img3[1]), axis=1)
    disp2 = np.concatenate((img3[2], img3[3]), axis=1)
    disp3 = np.concatenate((disp1, disp2), axis=0)

    disp1 = np.concatenate((img4[0], img4[1]), axis=1)
    disp2 = np.concatenate((img4[2], img4[3]), axis=1)
    disp4 = np.concatenate((disp1, disp2), axis=0)

    disp5 = np.concatenate((disp3, disp4), axis=1)

    disp = np.concatenate((img1, img2), axis=1)
    disp = np.concatenate((disp, disp5), axis=0)
    cv2.imwrite('./test/'+str(index)+'.jpg', disp)
    print('save img {}'.format(index))


def gps(query):
    rows = []
    img_name = query.split('/')[-1]
    with open('../../../washington-square1.csv') as f:
        fl = csv.reader(f)
        fields = next(fl)

        for row in fl:
            dic = {}
            for i in range(len(row)):
                dic[fields[i]] = row[i]
            rows.append(dic)
    
    for r in rows:
        if img_name in r['image']:
            gps = (float(r['snapped_lat']), float(r['snapped_lon']))
    
    dist_index = [(0, 1000) for i in range(8)]
    for i, r in enumerate(rows):
        if '2016' in r['image'] or '201701' in r['image']:
            if len(r['snapped_lat']) > 0:
                d = (float(r['snapped_lat'])-gps[0])**2 + (float(r['snapped_lon'])-gps[1])**2
            else:
                d = 1000
            if d < dist_index[-1][1]:
                dist_index[-1] = (i, d)
                dist_index.sort(key=lambda x: x[1])
    
    gps_imgs = []
    for i in range(8):
        path_list = rows[dist_index[i][0]]['image'].split('/')
        gps_img = '/data_1/washington-square/2/'+path_list[1][:6]+'/'+path_list[-1]
        gps_imgs.append(gps_img)
    # print(gps_img)
    # print(gps)
    # print(rows[index]['snapped_lat'], rows[index]['snapped_lon'])
    return gps_imgs


if __name__ == '__main__':
    path = 'result.txt'
    with open(path) as f:
        pairs = f.readlines()
    
    for i, pair in enumerate(pairs):
        print(i)
        s=pair.strip('\n').split(' ')
        test=s[0]
        train=s[1][:-1]
        mix(train, test, i)
        #if i == 166:
         #   break