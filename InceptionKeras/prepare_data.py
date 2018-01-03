import os
import scipy.io as sio
from shutil import copyfile

'''
This file is for preprocessing the data structure to make the keras 
image generator able to recognize the different classes
'''

def getLabelsAndPathsFromMatFile(path):
    mat_file = sio.loadmat(path)
    image_annos = mat_file['annotations'][0]
    classes = mat_file['class_names'][0]

    labels_and_paths = []

    for annotation in image_annos:
        labels_and_paths.append([annotation[0][0], classes[annotation[5][0][0]-1][0]])
    return labels_and_paths

#Save all differect labels with id as JSON
label_dic = {}
labels_and_paths = getLabelsAndPathsFromMatFile('data/cars_annos.mat')
for label in labels_and_paths:
    if label[1] not in label_dic:
        label_dic[label[1]] = [len(label_dic), 1]
    else:
        label_dic[label[1]][1] += 1
with open('label_dic.json', 'a') as the_file:
    the_file.write(str(label_dic))

#Create sub dirs for train and validation data
try:
    os.mkdir('./data/train_data/')
    os.mkdir('./data/train_data/images_per_class/')
    os.mkdir('./data/val_data/')
    os.mkdir('./data/val_data/images_per_class/')
except:
    print("Error while creating directories ")

#Create a sub folder for each label / class
try:
    for label in label_dic:
        os.mkdir('./data/train_data/images_per_class/' + str(label_dic[label][0]))
        os.mkdir('./data/val_data/images_per_class/' + str(label_dic[label][0]))
except:
    print("Error while creating subdirs for labels")



#Copy images to the subfolder
current_label = ""
counter = 0
count_per_label = 0
#Copy each image into a subdir which represents the class
for label in labels_and_paths:
    if current_label != label[1]:
        current_label = label[1]
        counter = 0
        count_per_label = label_dic[label[1]][1]
    #Dividing into training and validation
    if counter > count_per_label * 0.8:
        copyfile("./data/" + label[0], "./data/val_data/images_per_class/" + str(label_dic[label[1]][0]) + "/" + label[0].split("/")[1])
    else:
        copyfile("./data/" + label[0], "./data/train_data/images_per_class/" + str(label_dic[label[1]][0]) + "/" + label[0].split("/")[1])
    counter += 1