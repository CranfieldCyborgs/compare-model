from keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import numpy as np
from os import path

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

def plot(H, epochs):
    # plot the training loss and accuracy
    N = epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on COVID-19 Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('performance2.png', ppi=300)

def train(model, epochs, train_generator, test_generator, model_dest):
    # Saving parameters of each epoch
    # Todo: parametrize this or separate to function
    checkpoint_path = model_dest + "/cp-{epoch:04d}.ckpt"
    checkpoint_dir = path.dirname(checkpoint_path)
    print(checkpoint_dir)

    # Prepare callbacks for model saving and for learning rate adjustment.
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                #monitor='val_acc',
                                verbose=1,
                                #  period=5,
                                #save_best_only=True
                                save_weights_only=True
                                )


    # train the model
    print("[INFO] training head...")

    H = model.fit_generator(
            train_generator,
            steps_per_epoch=20,
            epochs=epochs,
            validation_data=test_generator,
            validation_steps=20,
            callbacks=[cp_callback]
            )

    return H