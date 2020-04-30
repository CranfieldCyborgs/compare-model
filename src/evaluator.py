import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

def evaluate(model, valid_X, valid_Y, train_generator, model_dest, model_figures_dest):
    # according to the performance curves choose the best parameters
    #! ls checkpoint_dir
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(model_dest + '/cp-0001.ckpt')

    # Using validation datasets to predict
    print("[INFO] evaluating network...")
    # predval = model.predict(valid_X)
    # for reccall
    predval = model.predict(valid_X)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predval2 = np.argmax(predval, axis=1)

    name = []
    for i in train_generator.class_indices.keys():
        name.append(i)
    # print(train_generator.class_indices) # name in dict
    print(classification_report(valid_Y.argmax(axis=1), predval2, target_names=name))


    # scores = model.evaluate(valid_X, valid_Y, verbose=1)
    scores = model.evaluate(valid_X, valid_Y, verbose=1)
    print('Validation loss:', scores[0])
    print('Validation accuracy:', scores[1])


    #plot AUC
    

    fig, c_ax = plt.subplots(1, 1,figsize=(8,8))
    for (i, label) in enumerate(train_generator.class_indices):
        fpr, tpr, thresholds = roc_curve(valid_Y[:, i].astype(int), predval[:, i])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (label, auc(fpr, tpr)))

    # Set labels for plot
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')

    # Todo: Change this to automatically get filename
    fig.savefig('./' + model_figures_dest + '/auc_ResNet50.png')


    # print("[INFO] saving COVID-19 detector model...")
    # model.save("model", save_format="h5")