import argparse
import sys

def run_training(evaluator, no_of_classes):
    import model_builder as mb
    from setup import models

    print("CompareModel training starting...")

    for index, base_model_and_params in enumerate(models):
        print("Training model [" + index + " of " + len(models) + "]")
        # DEBUG
        # if index in range(0, 5):
        #     continue

        (model_dest, _, base_keras_model) = base_model_and_params
        
        model = mb.construct(base_keras_model, no_of_classes)
        
        H = evaluator.train(model, model_dest)
        evaluator.plot(H, model_dest)

    print("CompareModel training completed.")


def run_evaluation(evaluator, no_of_classes, list_of_eval_models):
    import model_builder as mb
    from setup import models

    print("CompareModel evaluation starting...")

    # TODO: Make a list of all the optimum models and iterate over that too.
    for index, base_model_and_params in enumerate(models):
        print("Evaluating model [" + index + " of " + len(models) + "]")
        # DEBUG
        # if index in range(0, 5):
        #     continue

        (model_dest, model_figures_dest, base_keras_model) = base_model_and_params
        
        model = mb.construct(base_keras_model, no_of_classes)

        if list_of_eval_models is None:
            evaluator.eval_with(model, model_dest, model_figures_dest)
        else:
            # TODO Run with defaults if not enough files given
            if len(list_of_eval_models) != len(models):
                raise ValueError("Not enough models listed for evaluation.") 
            print("TODO")

    print("CompareModel evaluating completed.")


def run(option="train", list_of_eval_models = None):
    from setup import init, labels, EPOCHS
    from dataset import nih_image_path, nih_metadata_path, covid19_image_path, pneumonia_image_path
    from preprocessing import preprocessing
    import model_builder as mb
    from evaluator import Evaluator

    # test GPU and setup
    init()

    # Preprocessing
    (train_generator, test_generator, valid_X, valid_Y) = preprocessing(nih_metadata_path, nih_image_path, pneumonia_image_path, covid19_image_path, labels)

    no_of_classes = len(labels)

    evaluator = Evaluator(train_generator, test_generator, valid_X, valid_Y, EPOCHS)

    if option == "train":
        run_training(evaluator, no_of_classes)
    elif option == "evaluate":
        run_evaluation(evaluator, no_of_classes, list_of_eval_models)
    else:
        raise ValueError("Unrecognized run flag. Did you mean 'train' or 'evaluate'?")


parser = argparse.ArgumentParser()

parser.add_argument("-t", "--train", help="Run training", action="store_true")
parser.add_argument("-e", "--evaluate", help="Run evaluation", action="store_true")

parser.add_argument("-m", "--models", help="Specify list of stored model files for the evalutor to use for the evaluator. Otherwise the default file <TODO: filename> will be used.", nargs="+")

args = parser.parse_args()

# TODO Do some testing of the cmdline utility and functionality
if args.evaluate and args.models:
    # TODO convert them into paths and then pass it in?
    print(args.models)
    run("evaluate", args.models)
elif args.train and args.models:
    print("Program options not recognized.")
    parser.print_help()
    sys.exit()
elif args.train:
    print("Training...")
    run("train")
elif args.evaluate:
    print("Evaluating...")
    run("evaluate")
else:
    print("Program options not recognized.")
    parser.print_help()
    sys.exit()