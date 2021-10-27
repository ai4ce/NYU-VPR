# Authors: Dhruv Gaba and Purva Patel
# this code generates the database for the localization through VLAD and K-Means Codeook

# This code is responsible for the generation of the VLAD matrices and codebook
# for the given images dataset
# VLAD : Vector for locally aggregated Descriptors
# Codebook: is an object for the cluster of high dimensional features derived through the images

import glob
import itertools
import pickle
import os
from tqdm import tqdm
import csv
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

empty_image = []
VLAD_descriptors_original = list()
k_clusters = 32

# inputs
# getDescriptors for whole dataset
# Path = path to the image dataset
# functionHandleDescriptor={describeSURF, describeSIFT, describeORB}
def getDescriptors(path, functionHandleDescriptor, kmeans):
    print( "Start getDescriptors once" )
    descriptors = list()
    total_descriptors_lst = list()

    valid_img_path = path
    for imagePath in (path):
        im = cv2.imread(imagePath)
        # check if the image is not empty
        if im is not None:
            kp, des = functionHandleDescriptor(im)
            # if des is None:
            #     descriptors.append(np.zeros((1, 128)))
            if des is not None and des.all() != None:
                # descriptors.append(des)
                for point in des:
                    total_descriptors_lst.append(point)
            else:
                valid_img_path.remove(imagePath) 
                empty_image.append(imagePath)

        else:
            # add into the Empty Path list
            valid_img_path.remove(imagePath)
            print("Empty Path:",imagePath)
            empty_image.append(imagePath)
    
    kmeans = kMeansDictionary(total_descriptors_lst, k_clusters, kmeans)

    print( "End getDescriptors once" )
    # return descriptors, path
    return valid_img_path, kmeans

# input Descriptors array
# training = a set of descriptors
def kMeansDictionary(training, k, kmeans):
    # K-means algorithm
    print("Start kMeansDictionary")
    # est = KMeans(n_clusters=k, init='k-means++', tol=0.0001, verbose=1).fit(training)
    X = np.asarray(training)
    kmeans = kmeans.partial_fit(X)

    # centers = est.cluster_centers_
    # labels = est.labels_
    # est.predict(X)
    print("End kMeansDictionary")
    return kmeans
    # clf2 = pickle.loads(s)


# compute vlad descriptors for te whole dataset
# input: path = path of the dataset
#        functionHandleDescriptor={describeSURF, describeSIFT, describeORB}
#        visualDictionary = a visual dictionary from k-means algorithm
def getVLADDescriptors(path_lst, functionHandleDescriptor, visualDictionary):
    vlad_descriptors = list()
    for imagePath in tqdm(path_lst):
        # print(imagePath)
        im = cv2.imread(imagePath)
        kp, des = functionHandleDescriptor(im)
        if des is not None and des.all() != None:
            v = VLAD(des, visualDictionary)
            vlad_descriptors.append(v)

    return vlad_descriptors


# fget a VLAD descriptor for a particular image
# input: X = descriptors of an image (M x D matrix)
# visualDictionary = precomputed visual dictionary
def VLAD(X, visualDictionary):
    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels = visualDictionary.labels_
    k = visualDictionary.n_clusters

    m, d = X.shape
    V = np.zeros([k, d])
    # computing the differences

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels == i) > 0:
            # add the diferences
            V[i] = np.sum(X[predictedLabels == i, :] - centers[i], axis=0)

    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V) * np.sqrt(np.abs(V))
    # L2 normalization
    V = V / np.sqrt(np.dot(V, V))
    return V

# generates the surf detectors and descriptors
def describeSURF(image):
    surf = cv2.xfeatures2d.SURF_create(400, extended=True)
    # it is better to have this value between 300 and 500
    kp, des = surf.detectAndCompute(image, None)
    return kp, des

# load csv file
def load_csv():
    rows=[]
    with open('/home/deansheng/washington-square1.csv') as f:
        fl = csv.reader(f)
        fields = next(fl)
        for row in fl:
            dic = {}
            for i in range(len(row)):
                dic[fields[i]] = row[i]
            rows.append(dic)
    return rows
# find gps of the image
def find_path(image, rows):
    for r in rows:
        if image.split('/')[-1] in r['image']:
            path = "/data_1/washington-square/" + r['image']
            break
    return path

if __name__ == "__main__":
    print("Start")
    total_path = []
    # k_clusters is the number of clusters desired.
    kmeans = MiniBatchKMeans(n_clusters=k_clusters, init='k-means++', tol=0.0001)
    # computing the SURF descriptors, K_means codebook, and the VLAD descriptors
    # global total_descriptors_lst = list()
    #call getDescriptors muti-times

    rows = load_csv()

    for num in range(1, 17):
        print("Time:", num, end=" ")
        path_to_images_file = []
        # get paths from one image path file
        file_name = "/home/deansheng/test/vlad/separate_train_data/train_"+str(num)+".txt"
        f = open(file_name, 'r')
        one_path = f.readlines()
        for p in one_path:
            temp_path = find_path(p.split('\n')[0], rows)
            path_to_images_file.append(temp_path)
        f.close()
        # des_lst, valid_path = getDescriptors(path_to_images_file, describeSURF)
        valid_path, kmeans = getDescriptors(path_to_images_file, describeSURF, kmeans)
        total_path += valid_path

    # Saving the k_means_codeook in a text file
    file1 = open("/home/deansheng/test/vlad/original_k_means_codebook_object.txt", "wb")
    pickle.dump(kmeans, file1)
    file1.close()
    # call getVLADDescriptors 
    print("Start getVLADDescriptors(total_path, describeSURF, kmeans)")
    VLAD_descriptors_original = getVLADDescriptors(total_path, describeSURF, kmeans)
    # list to array
    VLAD_descriptors = np.asarray(VLAD_descriptors_original)
    desc_le = int(len(VLAD_descriptors[0]) / 128)
    print("Start np.reshape(VLAD_descriptors, (len(total_path), desc_le, 128))")
    vlad_database = np.reshape(VLAD_descriptors, (len(total_path), desc_le, 128))

    # reshaping the descriptos into a 2D vector for saving
    print("Start vlad_database.reshape(len(vlad_database) * k_clusters, 128)")
    updated_vlad_database = vlad_database.reshape(len(vlad_database) * k_clusters, 128)

    # saving the image names in a text file
    print("Start saving the image names in a text file")
    with open('/home/deansheng/test/vlad/original_img_list_original.txt', 'w') as f:
        for item in total_path:
            f.write("%s\n" % item)

    # free memory / del list 
    del vlad_database
    del kmeans
    del total_path

    # saving the vlad data base in an csv file
    print("Start saving the vlad data base in an csv file")
    write_f = open( "/home/deansheng/test/vlad/VLAD_database.csv",'w',encoding='utf-8')
    csv_writer = csv.writer(write_f)
    for item in updated_vlad_database:
        csv_writer.writerow(item)

    print("End")