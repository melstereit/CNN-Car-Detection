import os

import PIL.Image as Image
import scipy.io as sio
from tqdm import tqdm


class CarImages:

    def __init__(self):
        self.path_to_input_files = "./input/"
        self.mat_file = sio.loadmat(os.path.join(self.path_to_input_files, "cars_annos.mat"))

    def get_data(self, resize=None):
        """
        loading car images from the given input-path
        :param resize: if set to True images will be resized to the given standard size,
        if set to a tuple (width, height) the images will be resized to this size
        :return: 
        """
        paths, labels = self._get_paths_and_labels()
        images = self._load_images(paths, resize=resize)

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

    def _load_images(self, paths, resize=None):
        images = []
        for i in tqdm(range(len(paths))):
            path = os.path.join(self.path_to_input_files, paths[i])
            img = Image.open(path).convert('RGBA')
            if resize:
                if type(resize) is bool:
                    img = self.resize_image(image=img)
                elif type(resize) is tuple:
                    img = self.resize_image(image=img, size=resize)

            images.append(img)

        return images

    def load_image_sizes(self):
        # todo refactor to new class: ImageAnalyser
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

    @staticmethod
    def resize_image(image, size=(224, 168), fill_color=(0, 0, 0, 0)):
        width, height = size
        new_im = Image.new('RGBA', (width, height), fill_color)

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
        return new_im
