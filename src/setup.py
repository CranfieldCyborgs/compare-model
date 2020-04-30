# This supresses the debug log when using Tensorflow-gpu. This needs to be before the tf import.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from os.path import abspath, exists, basename

import tensorflow as tf
from keras.applications.resnet import ResNet50, ResNet101, ResNet152
from keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2


keras_models = [ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2]

model_destinations = ["ResNet50", "ResNet101", "ResNet152", "ResNet50v2", "ResNet101v2", "ResNet152v2"]

model_figures_destinations = list(map(lambda name: name+"_fig", model_destinations))


# models is an array of model, where
# model is tuple = (model_dest, model_figures_dest, kerasModel)
models = list(zip(model_destinations, model_figures_destinations, keras_models))

EPOCHS = 2

# Todo: make it work with the NIH-metadata
labels = ['Effusion', 'Atelectasis', 'Infiltration', 'Pneumonia', 'No Finding', 'COVID-19']

def gpu_test_and_setup():
    """
    This function executes and checks if the gpu version is configured correctly.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the fourth GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def setup_destinations(list_of_dest):
    """
    This function creates the required directories for the models and figures to be saved to.
    """
    paths = list(map(lambda filename: abspath(filename), list_of_dest))

    for path in paths:
        if not exists(path):
            print("Creating directory: " + basename(path))
            os.makedirs(path)
            print("Success in creating directory: " + basename(path))
        else:
            print("Directory already exists: " + basename(path))

def init():
    print("CompareModel initializing...")
    gpu_test_and_setup()
    setup_destinations(model_destinations)
    setup_destinations(model_figures_destinations)
    print("Initializing complete!")


