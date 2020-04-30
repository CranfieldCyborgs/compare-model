import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from keras.callbacks import ModelCheckpoint
from os import path

class Evaluator:
    def __init__(self, train_generator, test_generator, valid_X, valid_Y, epochs):
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.valid_X = valid_X
        self.valid_Y = valid_Y
        self.epochs = epochs
    
    def _train(self, model, model_dest):
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
                self.train_generator,
                steps_per_epoch=20,
                epochs=self.epochs,
                validation_data=self.test_generator,
                validation_steps=20,
                callbacks=[cp_callback]
                )

        return H

    def _plot(self, H):
        # plot the training loss and accuracy
        N = self.epochs
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

    def run_with(self, model, model_dest, model_figures_dest):
        
        H = self._train(model, model_dest)

        self._plot(H)
        
        # according to the performance curves choose the best parameters
        #! ls checkpoint_dir
        # latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(model_dest + '/cp-0001.ckpt')

        # Using validation datasets to predict
        print("[INFO] evaluating network...")
        # predval = model.predict(valid_X)
        # for reccall
        predval = model.predict(self.valid_X)

        # for each image in the testing set we need to find the index of the
        # label with corresponding largest predicted probability
        predval2 = np.argmax(predval, axis=1)

        name = []
        for i in self.train_generator.class_indices.keys():
            name.append(i)
        # print(train_generator.class_indices) # name in dict
        print(classification_report(self.valid_Y.argmax(axis=1), predval2, target_names=name))


        # scores = model.evaluate(valid_X, valid_Y, verbose=1)
        scores = model.evaluate(self.valid_X, self.valid_Y, verbose=1)
        print('Validation loss:', scores[0])
        print('Validation accuracy:', scores[1])


        #plot AUC
        

        fig, c_ax = plt.subplots(1, 1,figsize=(8,8))
        for (i, label) in enumerate(self.train_generator.class_indices):
            fpr, tpr, thresholds = roc_curve(self.valid_Y[:, i].astype(int), predval[:, i])
            c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (label, auc(fpr, tpr)))

        # Set labels for plot
        c_ax.legend()
        c_ax.set_xlabel('False Positive Rate')
        c_ax.set_ylabel('True Positive Rate')

        # Todo: Change this to automatically get filename
        fig.savefig('./' + model_figures_dest + '/auc_ResNet50.png')


        # print("[INFO] saving COVID-19 detector model...")
        # model.save("model", save_format="h5")