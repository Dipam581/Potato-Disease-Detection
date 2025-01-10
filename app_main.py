import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from tensorflow.keras.regularizers import l1, l2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class Prediction():

    def __init__(self):
        self.IMAGE_SIZE = 256
        self.BATCH = 32
        self.dataset = ""
        self.classes = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
        

    def fetch_dataset(self):
        if(not len(self.dataset)):
            self.dataset = tf.keras.preprocessing.image_dataset_from_directory(
                "PotatoDataset",image_size=(self.IMAGE_SIZE, self.IMAGE_SIZE), batch_size= self.BATCH, shuffle=True, verbose=True
            )
        self.split_dataset()
    

    def split_dataset(self):

        self.dataset = self.dataset.shuffle(1000)

        self.train_data = self.dataset.take(int(len(self.dataset)*0.8))
        print("Lenght of train dataset is-",len(self.train_data))

        remaining_data = self.dataset.skip(int(len(self.dataset)*0.8))

        self.test_data = remaining_data.take(int(len(remaining_data)*0.5))
        self.validation_data = remaining_data.skip(int(len(remaining_data)*0.5))

        print("Lenght of test dataset is-",len(self.test_data))
        print("Lenght of validation dataset is-",len(self.validation_data))

        self.preprocessed_data()


    def preprocessed_data(self):

        self.train_data = self.train_data.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
        self.test_data = self.test_data.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
        self.validation_data = self.validation_data.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)

        self.create_model()


    #Define model
    def create_model(self):
        model_name = "potato_disease_model.h5"
        if (os.path.exists(model_name)):
            model = load_model("potato_disease_model.h5")
            print("from saved model==============================")
            model.evaluate(self.test_data)

            self.prediction(model)
        else:
            #Rescaling images
            resize_and_rescaling = Sequential([
                tf.keras.layers.Resizing(255,255),
                tf.keras.layers.Rescaling(1.0/255),
            ])

            data_aug = Sequential([
                tf.keras.layers.RandomFlip('horizontal_and_vertical'),
                tf.keras.layers.RandomRotation(0.2),
            ])

            model = Sequential([
                resize_and_rescaling,
                data_aug,

                Conv2D(32, (3, 3), activation='relu', input_shape=(32, 256, 256, 3)),
                MaxPooling2D(pool_size=(2, 2)),

                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),

                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),

                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),

                # Flattening for fully connected layers
                Flatten(),
                Dense(3, activation='softmax')
            ])

            model.build(input_shape= (32, 256, 256, 3))
            model.summary()

            self.train_model(model)


    def train_model(self,model):

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            self.train_data,
            validation_data=self.validation_data,
            batch_size = 32,
            verbose = 1,
            epochs=10
        )
        self.visualize_and_predict_model(history, model)

    
    def visualize_and_predict_model(self, history, model):

        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        model.evaluate(self.test_data)

        self.prediction(model)


    def prediction(self, model):

        def predict(model, img):
            img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
            img_array = tf.expand_dims(img_array, 0)

            predictions = model.predict(img_array)

            predicted_class = self.classes[np.argmax(predictions[0])]
            confidence = round(100 * (np.max(predictions[0])), 2)
            return predicted_class, confidence
        
        plt.figure(figsize=(15, 15))
        for images, labels in self.test_data.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy())
                
                predicted_class, confidence = predict(model, images[i].numpy())
                actual_class = self.classes[labels[i]] 
                
                print(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
                
                # plt.axis("off")

        model.save("potato_disease_model.h5")  # Save the model in HDF5 format

    def load_saved_model(self):
        model_name = "potato_disease_model.h5"
        if (os.path.exists(model_name)):
            model = load_model("potato_disease_model.h5")



#Crtaeting instances of the class
pdp = Prediction()
pdp.fetch_dataset()

