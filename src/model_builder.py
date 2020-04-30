from keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

def construct(baseKerasModel):

    ## model part

    # load the ResNet50 network, ensuring the head FC layer sets are left off
    baseModel = baseKerasModel(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(128, 128, 3)))

    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(5, activation="softmax")(headModel) #How many classes?

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.5e-3), 
        loss='categorical_crossentropy', 
        metrics=['accuracy'])
    model.summary()

    return model