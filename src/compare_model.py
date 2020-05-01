import setup
from dataset import nih_image_path, nih_metadata_path, covid19_image_path, pneumonia_image_path
from preprocessing import preprocessing
import model_builder as mb
from evaluator import Evaluator

# test GPU and setup
setup.init()
EPOCHS = setup.EPOCHS

# Preprocessing
(train_generator, test_generator, valid_X, valid_Y) = preprocessing(nih_metadata_path, nih_image_path, pneumonia_image_path, covid19_image_path, setup.labels)

# model part

no_of_classes = len(setup.labels)

evaluator = Evaluator(train_generator, test_generator, valid_X, valid_Y, EPOCHS)

print("CompareModel training starting...")

for index, base_model in enumerate(setup.models):
    
    model_dest = base_model[0]
    model_figures_dest = base_model[1]
    base_keras_model = base_model[2]
    
    model = mb.construct(base_keras_model, no_of_classes)
    evaluator.train(model, model_dest)

print("CompareModel training completed.")

print("CompareModel evaluation starting...")

# TODO: Make a list of all the optimum models and iterate over that too.
for index, base_model in enumerate(setup.models):
    
    model_dest = base_model[0]
    model_figures_dest = base_model[1]
    
    evaluator.eval_with(model_dest, model_figures_dest)

print("CompareModel evaluating completed.")