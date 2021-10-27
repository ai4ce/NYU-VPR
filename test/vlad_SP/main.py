from superpoint import SuperPoint
import torch
import pickle
from tqdm import tqdm
import csv
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

k_clusters = 32
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
# input Descriptors array
# training = a set of descriptors
def kMeansDictionary(training, k, kmeans):
    # K-means algorithm
    X = np.asarray(training)
    kmeans = kmeans.partial_fit(X)
    print("End kMeansDictionary Once")
    return kmeans


def get_SP(img_paths, model, kmeans):
    descriptors = []
    for img in img_paths:
        pred = describe_SP(img, model)
        des = pred['descriptors'][0]
        des = torch.transpose(des, 0, 1)
        des = des.tolist() 
        for point in des:
            descriptors.append(point)

    kmeans = kMeansDictionary(descriptors, k_clusters, kmeans)

    print("End get_SP Once")
    return descriptors, kmeans


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def read_image(path, device):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None, None, None
    
    inp = frame2tensor(image, device)
    return image, inp


def describe_SP(img, model):
    model.eval()
    _, inp = read_image(img, device)
    pred = model({'image': inp})
    return pred


def getVLADDescriptors(path_lst, visualDictionary):
    print("Start getVLADDescriptors")
    vlad_descriptors = []
    for imagePath in tqdm(path_lst):
        pred = describe_SP(imagePath, model)
        des = pred['descriptors'][0]
        des = torch.transpose(des, 0, 1)
        des = des.cpu().detach().numpy()
        des = des.astype(np.float64)
        if des is not None:
            v = VLAD(des, visualDictionary)
            vlad_descriptors.append(v)
    print("End getVLADDescriptors")
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


if __name__ == '__main__':
    rows = load_csv()

    config = {}
    model = SuperPoint(config).to(device)
    kmeans = MiniBatchKMeans(n_clusters=k_clusters, init='k-means++', tol=0.0001)

    total_path = []
    
    file_name = "/home/deansheng/test/image_paths/train_image_paths.txt"
    f = open(file_name, 'r')
    one_path = f.readlines()
    for p in one_path:
        temp_path = find_path(p.split('\n')[0], rows)
        total_path.append(temp_path)
    f.close()

    for i in range(10000):
        if (i+1)*100 >= len(total_path):
            descriptors, kmeans = get_SP(total_path[i*100:], model, kmeans)
            break
        else:
            descriptors, kmeans = get_SP(total_path[i*100:(i+1)*100], model, kmeans)
        print("i=",i)


    # Saving the k_means_codeook in a text file
    file1 = open("/home/deansheng/test/vlad_SP/resource/SP_k_means_codebook_object.txt", "wb")
    pickle.dump(kmeans, file1)
    file1.close()
    # End saving k_means_codeook 

    # get VLAD_descriptors_original
    VLAD_descriptors_original = getVLADDescriptors(total_path, kmeans)
    VLAD_descriptors = np.asarray(VLAD_descriptors_original)
    desc_le = int(len(VLAD_descriptors[0]) / 256)
    vlad_database = np.reshape(VLAD_descriptors, (len(total_path), desc_le, 256))
    updated_vlad_database = vlad_database.reshape(len(vlad_database) * k_clusters, 256)

    # saving the image names in a text file
    print("Start saving the image names in a text file")
    with open('/home/deansheng/test/vlad_SP/resource/SP_img_list_original.txt', 'w') as f:
        for item in total_path:
            f.write("%s\n" % item)

    # free memory / del list 
    del vlad_database
    del VLAD_descriptors_original
    del VLAD_descriptors
    del kmeans
    del total_path

    # Start saving the vlad data base in an csv file
    print("Start saving the vlad data base in an csv file")
    write_f = open( "/home/deansheng/test/vlad_SP/resource/SP_vlad_database.csv",'w',encoding='utf-8')
    csv_writer = csv.writer(write_f)
    for item in updated_vlad_database:
        csv_writer.writerow(item)
    print("End")

