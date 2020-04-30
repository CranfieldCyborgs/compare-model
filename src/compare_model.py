import setup
import dataset as ds
from preprocessing import preprocessing
import trainer
import evaluator

# test GPU and setup
setup.gpu_test_and_setup()


# Overall variables
nih_metadata_path = ds.nih_metadata_path
nih_image_path = ds.nih_image_path

pneumonia_image_path = ds.pneumonia_image_path
covid19_image_path = ds.covid19_image_path

model_dest = setup.models[0][0]
model_figures_dest = setup.models[0][1]

baseKerasModel = setup.models[0][2]

# End of global variables

# Preprocessing
(train_generator, test_generator, valid_X, valid_Y) = preprocessing(nih_metadata_path, nih_image_path, pneumonia_image_path, covid19_image_path)


# ## model part

model = trainer.construct(baseKerasModel)
EPOCHS = setup.EPOCHS

H = trainer.train(model, EPOCHS, train_generator, test_generator, model_dest)

trainer.plot(H, EPOCHS)


evaluator.evaluate(model, valid_X, valid_Y, train_generator, model_dest, model_figures_dest)