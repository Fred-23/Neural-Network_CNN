# from keras.models import Sequential

# my_VGG16 = Sequential()  # Création d'un réseau de neurones vide 

# from keras.layers import Conv2D, MaxPooling2D

# my_VGG16 = Sequential()  # Création d'un réseau de neurones vide 

# # Ajout de la première couche de convolution, suivie d'une couche ReLU
# my_VGG16.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3), padding='same', activation='relu'))

# # Ajout de la deuxième couche de convolution, suivie  d'une couche ReLU
# my_VGG16.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

# # Ajout de la première couche de pooling
# my_VGG16.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# #Implementation of an VGG-16
# import tensorflow as tf,keras
# #import tensorflow_addons as tfa
# #from tensorflow import keras 
# #from PIL import Image, ImageOps, ImageFilter
# from keras.applications.vgg16 import VGG16
# from keras.preprocessing.image import load_img, img_to_array
# from keras.applications.vgg16 import preprocess_input
# model = VGG16() # Création du modèle VGG-16 implementé par Keras
# img = load_img('cat.jpg', target_size=(224, 224))  # Charger l'image
# img = img_to_array(img)  # Convertir en tableau numpy
# img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))  # Créer la collection d'images (un seul échantillon)
# img = preprocess_input(img)  # Prétraiter l'image comme le veut VGG-16

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
            

my_VGG16 = Sequential()  # Création d'un réseau de neurones vide 

# Ajout de la première couche de convolution, suivie d'une couche ReLU
my_VGG16.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3), padding='same', activation='relu'))

# Ajout de la deuxième couche de convolution, suivie  d'une couche ReLU
my_VGG16.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

# Ajout de la première couche de pooling
my_VGG16.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


my_VGG16.add(Flatten())  # Conversion des matrices 3D en vecteur 1D

# Ajout de la première couche fully-connected, suivie d'une couche ReLU
my_VGG16.add(Dense(4096, activation='relu'))

# Ajout de la deuxième couche fully-connected, suivie d'une couche ReLU
my_VGG16.add(Dense(4096, activation='relu'))

# Ajout de la dernière couche fully-connected qui permet de classifier
my_VGG16.add(Dense(1000, activation='softmax'))