

import PIL.Image as Image
import scipy.io as sio
import os

path_to_input_files = "./input/"
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

def loadImages(paths):
    images = []
    image_width = []
    for i in range(len(paths)):
        path = os.path.join(path_to_input_files, paths[i])
        img = Image.open(path).convert('RGBA')
        width, height = img.size
        image_width.append(width)
        print("Image ", len(image_width), "done")

    return image_width

inputPaths, classes = getData()
images = loadImages(inputPaths)

print(max(images))
print(min(images))