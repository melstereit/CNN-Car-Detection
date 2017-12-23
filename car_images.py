import os

import PIL.Image as Image
import scipy.io as sio
from tqdm import tqdm


class CarImages:

    def __init__(self):
        self.path_to_input_files = "./input/"
        self.mat_file = sio.loadmat(os.path.join(self.path_to_input_files, "cars_annos.mat"))

    def get_data(self):
        paths, labels = self._get_paths_and_labels()
        images = self._load_images(paths)

        return images, labels

    def _get_paths_and_labels(self):
        image_annos = self.mat_file['annotations'][0]
        classes = self.mat_file['class_names'][0]

        labels = []
        paths = []

        for annotation in image_annos:
            labels.append(classes[annotation[5][0][0] - 1][0])
            paths.append(annotation[0][0])

        return paths, labels

    def _load_images(self, paths):
        images = []
        for i in tqdm(range(len(paths))):
            path = os.path.join(self.path_to_input_files, paths[i])
            img = Image.open(path).convert('RGBA')
            images.append(img)

        return images

    def load_image_sizes(self):
        paths, _ = self._get_paths_and_labels()
        image_sizes_str = []
        image_sizes = []
        heights = []
        widths = []
        for i in tqdm(range(len(paths))):
            path = os.path.join(self.path_to_input_files, paths[i])
            img = Image.open(path).convert('RGBA')
            image_sizes_str.append(str(img.size))
            image_sizes.append(img.size)
            height, width = img.size
            heights.append(height)
            widths.append(width)

        return image_sizes_str, image_sizes, heights, widths
