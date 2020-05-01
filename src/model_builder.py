from keras.layers import AveragePooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense, Input
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

def construct(base_keras_model, no_of_classes):
    base_model = base_keras_model(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(128, 128, 3)))

    # construct the head of the model that will be placed on top of the
    # the base model
    head_model = base_model.output

    # Changing the averagepooling method for inceptionv3
    if base_keras_model == InceptionV3:
        head_model = GlobalAveragePooling2D()(head_model)
        head_model = Dense(512, activation="relu")(head_model)
        head_model = Dropout(0.5)(head_model)
        head_model = Dense(512, activation="relu")(head_model)
    else:
        head_model = AveragePooling2D(pool_size=(4, 4))(head_model)
        head_model = Flatten(name="flatten")(head_model)
        head_model = Dense(64, activation="relu")(head_model)

    head_model = Dropout(0.5)(head_model)
    head_model = Dense(no_of_classes, activation="softmax")(head_model) #How many classes?

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=base_model.input, outputs=head_model)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.5e-3), 
        loss='categorical_crossentropy', 
        metrics=['accuracy'])
    model.summary()

    return model