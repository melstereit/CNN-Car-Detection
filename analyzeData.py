import scipy.io as sio


def getLabelsAndPathsFromMatFile(path):
    mat_file = sio.loadmat(path)
    image_annos = mat_file['annotations'][0]
    classes = mat_file['class_names'][0]

    labels_and_paths = []

    for annotation in image_annos:
        labels_and_paths.append((annotation[0][0], classes[annotation[5][0][0]-1][0]))
    return labels_and_paths

labels_and_paths = getLabelsAndPathsFromMatFile('input/cars_annos.mat')

print(labels_and_paths)