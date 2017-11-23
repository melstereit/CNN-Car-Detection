import scipy.io as sio
from urllib import request, parse
from htmldom import htmldom

def getLabelsAndPathsFromMatFile(path):
    mat_file = sio.loadmat(path)
    image_annos = mat_file['annotations'][0]
    classes = mat_file['class_names'][0]

    labels_and_paths = []

    for annotation in image_annos:
        labels_and_paths.append([annotation[0][0], classes[annotation[5][0][0]-1][0]])
    return labels_and_paths

def getAllCarBrands():
    f = request.urlopen('https://www.kfz-mag.de/marken-modelle')
    content = str(f.read(), 'utf-8')
    html_dom = htmldom.HtmlDom().createDom(content)
    brands = html_dom.find('.articleColumn .brand h3')
    brands_array = []
    for brand in brands:
        brands_array.append(brand.text().lower())
    brands_array.append("volkswagen")
    brands_array.append("spyker")
    brands_array.append("scion")
    brands_array.append("plymouth")
    brands_array.append("mclaren")
    brands_array.append("geo")
    brands_array.append("eagle")
    brands_array.append("am general")
    brands_array.append("hummer")


    return brands_array

brands = getAllCarBrands()
labels_and_paths = getLabelsAndPathsFromMatFile('input/cars_annos.mat')

#Just get the car brand and add in to each image in labels_and_paths
for image in labels_and_paths:
    possible_brands = []
    for brand in brands:
        if brand in image[1].lower():
            possible_brands.append(brand)
    image.append(max(possible_brands, key=len))

#Count the number of images for each brand
brands_in_data = []
brands_count = []
for image in labels_and_paths:
    if(image[2] not in brands_in_data):
        brands_in_data.append(image[2])
        brands_count.append(1)
    else:
        brands_count[brands_in_data.index(image[2])] += 1
for brand in range(len(brands_in_data)):
    print(brands_in_data[brand], brands_count[brand])