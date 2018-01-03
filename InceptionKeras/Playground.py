import keras
from keras.applications import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import LambdaCallback
from keras.engine import Model
from keras.layers import Dense, GlobalMaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import requests

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    "./data/train_data/images_per_class/",
    target_size=(299, 299),
    batch_size=10
)

validation_generator = test_datagen.flow_from_directory(
    "./data/val_data/images_per_class/",
    target_size=(299, 299),
    batch_size=10
)

batch_counter = 0

def send_mail_with_progress(batch, batch_counter):
    batch_counter += 1
    print("Mail sent")
    if batch_counter % 10 is 1:
        content = "Model is in training. Computed" + batch_counter + " batches of 10 images."
        req  = "https://api.elasticemail.com/v2/email/send?apikey=31abeb50-eff0-4863-a167-b7f1c8689c83&subject=Python info&from=meinemail@kruegerbiz.de&to=marvin.krueger@yahoo.de&msgTo=marvin.krueger@yahoo.de&bodyHtml=" + content + "&encodingType=0&isTransactional=True"
        requests.get(req)

train_progress_mail_callback = LambdaCallback(
    on_batch_begin=lambda batch,logs: send_mail_with_progress(batch, batch_counter))

def initializing_and_train_top_layer():
    base_model = InceptionV3(weights='imagenet', include_top=False)

    #Freeze the layers weights
    for layer in base_model.layers:
        layer.trainable = False

    #Include top layer
    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(196, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    #Fit
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=13096,
        nb_epoch=2,
        callbacks=[train_progress_mail_callback],
        validation_data=validation_generator,
        nb_val_samples=3089,
        class_weight='auto')

    #Save
    model.save("model.keras")

def train_top_layers():
    layers_to_freeze = 172
    model = keras.models.load_model('model.keras')
    for layer in model.layers[:layers_to_freeze]:
        layer.trainable = False
    for layer in model.layers[layers_to_freeze:]:
        layer.trainable = True
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #Fit
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=13096,
        nb_epoch=2,
        callbacks=[train_progress_mail_callback],
        validation_data=validation_generator,
        nb_val_samples=3089,
        class_weight='auto')

    #Save
    model.save("model.keras")

train_top_layers()