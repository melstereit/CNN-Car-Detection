import os
import numpy as np
import scipy.io as sio
import PIL.Image as Image
import logging

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pathlib import Path
from tqdm import tqdm


class ReadImages:
    logging.basicConfig(filename='log/info.log', level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.basicConfig()

    def __init__(self, data_path="./data"):
        self.DATA_PATH = data_path
        self.DATA_SIZE = 128
        if not os.path.exists("models"):
            os.makedirs("models")
        if not os.path.exists("np_data"):
            os.makedirs("np_data")

    def _get_labels_and_classes(self):
        label_encoder = preprocessing.LabelEncoder()
        mat_file = sio.loadmat(self.DATA_PATH + "/cars_annos.mat")
        image_annos = mat_file['annotations'][0]

        size = len(image_annos)
        classes = mat_file['class_names'][0]
        y = np.zeros((size, len(classes)), dtype=int)
        paths = []
        label_encoder.fit(classes)
        logging.info("Read annotations and label")
        for i, annotation in tqdm(enumerate(image_annos)):
            paths.append(annotation[0][0])
            j = label_encoder.transform([classes[annotation[5][0][0] - 1][0]])
            y[i][j] = 1
        return paths, y

    def _get_labels_and_classes_test_version(self):
        mat_file = sio.loadmat(self.DATA_PATH + "/cars_annos.mat")
        image_annos = mat_file['annotations'][0]

        #Size, for testing 120
        size = 99
        y = np.zeros((100, 2))
        y[:50][0] = 1
        y[50:][1] = 1
        paths = []

        for i, annotation in enumerate(image_annos):
            paths.append(annotation[0][0])
            if i == size:
                break
        return paths, y

    def get_train_and_test_data(self):
        logging.info("Start preprocessing and training")
        x_file = Path("np_data/X.npy")
        y_file = Path("np_data/Y.npy")
        d_size = self.DATA_SIZE
        if x_file.is_file() and y_file.is_file():
            logging.info("Data already processed")
            X = np.load(x_file)
            y = np.load(y_file)
        else:
            logging.info("Processing data")
            paths, y = self._get_labels_and_classes()
            X = np.zeros((len(y), d_size, d_size, 3))
            logging.info("Get train and test data")
            for i, path in tqdm(enumerate(paths)):
                img = Image.open(self.DATA_PATH + "/" + path).convert('RGB')

                X[i] = self._resize_image(img, size=(d_size, d_size))
            np.save("np_data/X.npy", X)
            np.save("np_data/Y.npy", y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        return (X_train, y_train), (X_test, y_test)

    def _resize_image(self, image, size=(128, 128), fill_color=(0, 0, 0)):
        width, height = size
        new_im = Image.new('RGB', (width, height), fill_color)

        ratio_target = height/width
        width_actual, height_actual = image.size
        ratio_actual = height_actual/width_actual

        if ratio_target == ratio_actual:
            # image already has right ratio
            # width and height can be resized equally
            img_resized = image.resize(size=(int(width), int(height)))
        elif ratio_target > ratio_actual:
            # image is wider then target
            # image height has to be computed new
            factor = width / width_actual
            height_target = height_actual * factor
            img_resized = image.resize(size=(int(width), int(height_target)))
        elif ratio_target < ratio_actual:
            # image is higher then target
            # image width has to be computed new
            factor = height / height_actual
            width_target = width_actual * factor
            img_resized = image.resize(size=(int(width_target), int(height)))
        else:
            raise Exception("Something went wrong")
        new_im.paste(img_resized)
        img = np.asarray(new_im)
        return img
