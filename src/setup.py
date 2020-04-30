# This supresses the debug log when using Tensorflow-gpu. This needs to be before the tf import.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from keras.applications.resnet import ResNet50, ResNet101, ResNet152
from keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2


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


keras_models = [ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2]

model_destinations = ["ResNet50", "ResNet101", "ResNet152", "ResNet50v2", "ResNet101v2", "ResNet152v2"]

model_figures_destinations = list(map(lambda name: name+"_fig", model_destinations))


# models is an array of model, where
# model is tuple = (model_dest, model_figures_dest, kerasModel)
models = list(zip(model_destinations, model_figures_destinations, keras_models))

EPOCHS = 2