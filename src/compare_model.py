import setup
from dataset import nih_image_path, nih_metadata_path, covid19_image_path, pneumonia_image_path
from preprocessing import preprocessing
import model_builder as mb
from evaluator import Evaluator

# test GPU and setup
setup.init()

# Overall variables
model_dest = setup.models[0][0]
model_figures_dest = setup.models[0][1]

baseKerasModel = setup.models[0][2]
EPOCHS = setup.EPOCHS

# Preprocessing
(train_generator, test_generator, valid_X, valid_Y) = preprocessing(nih_metadata_path, nih_image_path, pneumonia_image_path, covid19_image_path, setup.labels)


# ## model part

# Todo: len(setup.labels) - 1 because COVID-19 is not a class yet
no_of_classes = len(setup.labels)
model = mb.construct(baseKerasModel, no_of_classes)

evaluator = Evaluator(train_generator, test_generator, valid_X, valid_Y, EPOCHS)

evaluator.run_with(model, model_dest, model_figures_dest)