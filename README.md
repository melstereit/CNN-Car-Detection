# CNN-Car-Detection
A Convolutional Neural Network for categorizing cars

picture-size für das netzwerk: 224x168

## Data
Source --> https://www.google.de/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwiwy5-X197YAhVCKFAKHXWGDcwQFggtMAA&url=http%3A%2F%2Fai.stanford.edu%2F~jkrause%2Fcars%2Fcar_dataset.html&usg=AOvVaw2qWV70n1td3vyAGhGCK0JS
about 16.0000 car images. Each labeled with a car model. 196 different car model labels at all. 

## Transfer Learning approach 
Inception V3 base model was used to categorize the images. 
Dataset was divided into 80 per train and 20 per val data. 
Used Xavier initializer for weight initialization. 

## Own model approach
