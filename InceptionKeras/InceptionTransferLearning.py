import keras
from keras.applications import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import LambdaCallback
from keras.engine import Model
from keras.layers import Dense, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import requests

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    "./data/train_data/images_per_class/",
    target_size=(299, 299),
    batch_size=10,
    class_mode='categorical'
)
def save_preview():
    # check out this
    #https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    #--> See which type of data the datagen will make out of the raw data
    i = 0
    for batch in train_datagen.flow_from_directory("./data/train_data/images_per_class/", target_size=(299, 299), batch_size=10,
                                                   save_to_dir='preview', save_prefix='car', save_format='jpeg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely



validation_generator = test_datagen.flow_from_directory(
    "./data/val_data/images_per_class/",
    target_size=(299, 299),
    batch_size=10,
    class_mode='categorical'
)

batch_counter = 0

def send_mail_with_text(text):
    print("Mail sent")
    content = text
    req  = "https://api.elasticemail.com/v2/email/send?apikey=31abeb50-eff0-4863-a167-b7f1c8689c83&subject=Python info&from=meinemail@kruegerbiz.de&to=marvin.krueger@yahoo.de&msgTo=marvin.krueger@yahoo.de&bodyHtml=" + content + "&encodingType=0&isTransactional=True"
    requests.get(req)

'''train_progress_mail_callback = LambdaCallback(
    on_batch_begin=lambda batch,logs: send_mail_with_progress(batch))'''

def initializing_and_train_top_layer():
    base_model = InceptionV3(weights='imagenet', include_top=False)

    #Freeze the layers weights
    for layer in base_model.layers:
        layer.trainable = False

    #Include top layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_initializer='glorot_normal', bias_initializer='zeros')(x)
    x = Dense(500, activation='relu', kernel_initializer='glorot_normal', bias_initializer='zeros')(x)
    predictions = Dense(196, activation='softmax', kernel_initializer='glorot_normal', bias_initializer='zeros')(x)

    model = Model(input=base_model.input, output=predictions)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    #Fit
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=13096,
        nb_epoch=2,
        validation_data=validation_generator,
        nb_val_samples=3089,
        class_weight='auto')

    #Save
    model.save("model.keras")

def train_top_layers():
    # --> https://www.kaggle.com/abnera/transfer-learning-keras-xception-cnn
    # If you want to train the last Two blocks of an Inception model it should be 172
    # layers before this number will used the pre-trained weights, layers above and including this number
    # will be re-trained based on the new data.
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
        validation_data=validation_generator,
        nb_val_samples=3089,
        class_weight='auto')

    #Save
    model.save("model.keras")
    return history


initializing_and_train_top_layer()
history_obj = train_top_layers()

send_mail_with_text("Calculation completed! Accuracy: " + history_obj.history['acc'])