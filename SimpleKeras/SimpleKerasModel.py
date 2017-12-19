import keras
import csv

import PIL.Image as Image
import scipy.io as sio
import os
import numpy as np

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D

path_to_input_files = "../input/"
def getData():
    mat_file = sio.loadmat(os.path.join(path_to_input_files, "cars_annos.mat"))

    image_annos = mat_file['annotations'][0]
    classes = mat_file['class_names'][0]

    labels = []
    paths = []

    for annotation in image_annos:
        labels.append(classes[annotation[5][0][0]-1][0])
        paths.append(annotation[0][0])
    return paths, labels


def convertClasses(classes):
    '''Convert from car classes to numbers'''
    brands = []

    with open('../brands.csv', 'r') as f:
        reader = csv.reader(f)
        brands = list(reader)
    brands = brands[0]
    availiable_brands = []
    #Add all avaiable brands in dataset
    for car_class in classes:
        for brand in brands:
            if brand in car_class.lower() and brand not in availiable_brands:
                availiable_brands.append(brand)


    for i in range(len(classes)):
        for brand_index in range(len(availiable_brands)):
            if availiable_brands[brand_index] in classes[i].lower():
                classes[i] = brand_index
                break

    for class_nr in range(len(classes)):
        class_vec = np.zeros(len(classes))
        class_vec[class_nr] = 1
        classes[class_nr] = class_vec

    return classes, availiable_brands

def loadImages(paths):
    images = []
    for i in range(len(paths)):
        path = os.path.join(path_to_input_files, paths[i])
        img = Image.open(path).convert('RGBA')
        arr = np.array(img)
        images.append(arr)
        print("Image ", len(images), "done")

    images = np.array(images)
    return images

inputPaths, classes = getData()
classes, brands = convertClasses(classes)
images = loadImages(inputPaths[:30])


modelK = Sequential([
    Conv2D(15, (9, 9), activation="relu", input_shape=images[0].shape, kernel_initializer='random_uniform'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(20, (9, 9), activation="relu", input_shape=images[0].shape, kernel_initializer='random_uniform'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(1000, activation='relu', kernel_initializer='random_uniform'),
    Dense(500, activation='relu', kernel_initializer='random_uniform'),
    Dense(250, activation='relu', kernel_initializer='random_uniform'),
    Dense(len(classes), activation='softmax', kernel_initializer='random_uniform')
])

modelK.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])

modelK.fit(np.array([images]), np.array(classes[:30]), epochs=5, batch_size=100)