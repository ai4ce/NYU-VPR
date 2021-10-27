import glob
import pickle
import argparse
import os
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import csv

k_clusters = 32

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


# finds the vlad for the query images with the help form the k means codebook: visaul dictionary
def getting_VLAD_for_Query_image(path, visualDictionary):
    descriptor = list()
    idImage = list()
    imagePaths = path

    for imagePath in tqdm(imagePaths):
        # print(imagePath)
        im = cv2.imread(imagePath)
        if im is not None:
            kp, des = describeSURF(im)
            if des is not None and des.all() != None:
                v = VLAD(des, visualDictionary)
                descriptor.append(v)
                idImage.append(imagePath)
    # list to array

    descriptor = np.asarray(descriptor)
    #print("descriptor:",descriptor)
    desc_le = int(len(descriptor[0]) / 128)
    #print("desc_le:",descriptor.shape)
    #print("len(idImage):",len(idImage))
    descriptor = np.reshape(descriptor, (len(idImage), desc_le, 128))

    return descriptor, idImage


# finds the top closest images from the database
def finding_closest_image_from_database(descriptors, descriptor):
    D = list()  # list to store the differnece of distance
    for image in descriptor[0]:  # vlad for query images
        for images in descriptors[0]:  # vlad for database images
            a = images
            b = image
            dis = np.linalg.norm(a - b)  # finding distance between querry and all images in database
            D.append(dis)

    D = np.reshape(D, (len(descriptor[1]), len(descriptors[1])))  # reshaping the distance array to queery*database
    D = np.array(D).tolist()
    temp_list = np.zeros((len(descriptor[1]), 5))  # making a temp list to store the top 5 matches from db to the querry
    temp_list = temp_list.tolist()
    list_of_indx = np.zeros(
        (len(descriptor[1]), 5))  # finding the index of the top 5 and storing to retrive the image name from db list
    list_of_b_indx = []

    for split in range(len(descriptor[1])):
        for x in range(5):
            temp_list[split][x] = sorted(D[split])[x]  # stores top 5 (least diff btw querry and db)
        for y in range(5):
            list_of_indx[split][y] = D[split].index(
                temp_list[split][y])  # find the index of top 5 images in the original D diff list
        for w in range(5):
            list_of_b_indx.append(descriptors[1][int(list_of_indx[split][
                                                         w])])  # with the index found previous it finds the respective image tag and stores it

    return list_of_b_indx


# generates the surf detectors and descriptors
def describeSURF(image):
    surf = cv2.xfeatures2d.SURF_create(400, extended=True)
    # it is better to have this value between 300 and 500
    kp, des = surf.detectAndCompute(image, None)
    return kp, des


def call_once(path_from_test_file, k_means_codebook_object, vlad_descriptors):
    print("Start Once")
    # start getting_VLAD_for_Query_image
    query_vlad = getting_VLAD_for_Query_image(path_from_test_file, k_means_codebook_object)
    # free memory
    """
    del k_means_codebook_object
    del path_from_test_file
    del vlad_database
    """
    # Start finding_closest_image_from_database
    nearest_images = finding_closest_image_from_database(vlad_descriptors, query_vlad)

    # saves list of closets images from database
    print("Start saves list of closets images from database")
    # write_file_name = "/home/deansheng/test/vlad/random_find_closest.txt"
    write_file_name = "/home/deansheng/test/vlad/VLAD_top_5_result.txt"
    with open(write_file_name, 'a') as nearest_image_file:   
        number_of_images = len(query_vlad[1])
        for j in range(number_of_images):
            # store original image + top 5
            pic_q = query_vlad[1][j]
            pic1 = nearest_images[(5 * j) + 0]
            pic2 = nearest_images[(5 * j) + 1]
            pic3 = nearest_images[(5 * j) + 2]
            pic4 = nearest_images[(5 * j) + 3]
            pic5 = nearest_images[(5 * j) + 4]
            one_line = pic_q+" "+pic1+" "+pic2+" "+pic3+" "+pic4 + " "+pic5+ "\n"
            nearest_image_file.write(one_line)
    print("END Once") 
    

if __name__ == '__main__':
    # reading  codebook saved from database code
    infile = open("/home/deansheng/test/vlad/original_k_means_codebook_object.txt", 'rb')
    k_means_codebook_object = pickle.load(infile)
    infile.close()

    # reading the train list of images
    print("Start reading the original list of images path")
    file2 = open("/home/deansheng/test/vlad/original_img_list_original.txt", "r")
    img_list = file2.read().split('\n')
    img_list.pop(-1)

    # reading vlad from the database
    print("Start reading vlad from the database(csv file)")
    csv_file=open('/home/deansheng/test/vlad/VLAD_database.csv')
    csv_reader_lines = csv.reader(csv_file)
    row_idx = 0
    col_idx = 0
    temp_mat = np.zeros((620864, 128), dtype=float)
    for one_line in csv_reader_lines:
        col_idx = 0
        for col_ele in one_line:
            temp_mat[row_idx][col_idx] = float(col_ele)
            col_idx += 1
        row_idx += 1
    csv_file.close()
    # reshaping the vald matrix after reading it from the file
    vlad_database = temp_mat.reshape((int(len(temp_mat) / k_clusters)), k_clusters, 128)

    # making a tuple of the images and vlad matrices
    vlad_descriptors = (vlad_database, img_list)  # tuple of vlad and img list

    # get test image paths
    file_name = "/home/deansheng/test/image_paths/test_image_paths.txt"
    f = open(file_name, 'r')
    one_path = f.readlines()
    path_from_test_file = list()
    for p in one_path:
        path_from_test_file.append(p.split('\n')[0])
    f.close()

    for i in range(20):
        if (i+1)*1000 >= len(path_from_test_file):
            call_once(path_from_test_file[i*1000:], k_means_codebook_object, vlad_descriptors)
            break 
        else:
            call_once(path_from_test_file[i*1000:(i+1)*1000], k_means_codebook_object, vlad_descriptors)
            print("Check...")

    print("END")
