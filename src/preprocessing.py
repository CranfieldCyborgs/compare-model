from os import path
from sklearn.model_selection import train_test_split
from glob import glob
import pandas as pd
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator

def generate_training_and_testing_set(nih_metadata_path, nih_image_path, pneumonia_image_path, covid19_image_path, labels):
    """
    There are 3 datasets.
    1. NIH datasets: contains more than 100,000 images of different thoracic disease.
    2. kaggle pneumonia datasets: contain 3800 pneumonia images and 1342 normal ones
    3. COVID 19: about 400 COVID-19 images collected by Ali & Jiaqi 
    """
    ### Part 1: data preprocessing

    ## 1.1 NIH data preprocessing 
    # read NIH data, the csv
    df_NIH = pd.read_csv(nih_metadata_path)
    # only keep the illnesss label and image names
    df_NIH = df_NIH[['Image Index', 'Finding Labels']]

    # create a column to store the full path of images for later reading
    # You should change the below path when running at your computer
    my_glob = glob(nih_image_path)
    print('Number of Observations: ', len(my_glob))
    full_img_paths = {path.basename(x): x for x in my_glob}
    df_NIH['path'] = df_NIH['Image Index'].map(full_img_paths.get)

    df_NIH = df_NIH[df_NIH['Finding Labels'].isin(labels)]


    ## 1.2 Kaggel pneumonia data preprocessing
    # add 3800 pneumonia images from kaggle https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
    glob2 = glob(pneumonia_image_path)
    df_extraPhe = pd.DataFrame(glob2, columns=['path'])
    df_extraPhe['Finding Labels'] = 'Pneumonia'


    ## 1.3 COVID-19 images preprocessing
    glob3 = glob(covid19_image_path)
    df_COVID19 = pd.DataFrame(glob3, columns=['path'])
    df_COVID19['Finding Labels'] = 'COVID-19' # be careful about the label here

    # concat the NIH pneumonia, kaggle pneumonia and COVID-19 images together
    # here is the final data set
    xray_data = pd.concat([df_NIH, df_extraPhe, df_COVID19])

    # calculate the number of each labels
    num = []
    for i in labels:
        temp = len(xray_data[xray_data['Finding Labels'].isin([i])])
        num.append(temp)

    print("DEBUG: Number of each label = ")
    print(dict(zip(labels, num)))

    # draw the data distribution
    df_draw = pd.DataFrame(data={'labels':labels, 'num':num})
    df_draw = df_draw.sort_values(by='num', ascending=False)
    ax = sns.barplot(x='num', y='labels', data=df_draw, color="green")
    # fig = ax.get_figure()
    # fig.savefig('./fig/a.png')

    # split data into train, test, validation
    train_set, valid_set = train_test_split(xray_data, test_size = 0.02, random_state = 42)

    train_set, test_set = train_test_split(train_set, test_size = 0.2, random_state = 8545)


    return (train_set, test_set, valid_set, xray_data)

def debug_set(train_set, test_set, valid_set, xray_data):
    """
    quick check to see that the training and test set were split properly
    """

    print("train set:", len(train_set))
    print("test set:", len(test_set))
    print("validation set:", len(valid_set))
    print('full data set: ', len(xray_data))

def data_gen_parameters():
    """
     Create ImageDataGenerator, to perform significant image augmentation
     Utilizing most of the parameter options to make the image data even more robust
     return just the randomly transformed data.
    """
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)
    return (train_datagen, test_datagen)

def data_generator(train_datagen, test_datagen, train_set, test_set, valid_set):


    # not read images directly. read the image name and then read images
    image_size = (128, 128) # image re-sizing target

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_set,
        directory=None,
        x_col='path', 
        y_col = 'Finding Labels',
        target_size=image_size,
        color_mode='rgb',
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_set,
        directory=None,
        x_col='path', 
        y_col = 'Finding Labels',
        target_size=image_size,
        color_mode='rgb',
        batch_size=64,
        class_mode='categorical'	
    )

    # create validation data generator
    valid_X, valid_Y = next(test_datagen.flow_from_dataframe(
        dataframe=valid_set,
        directory=None,
        x_col='path', 
        y_col = 'Finding Labels',
        target_size=image_size,
        color_mode='rgb',
        batch_size= len(valid_set),
        class_mode='categorical'
    ))

    return (train_generator, test_generator, valid_X, valid_Y)

def preprocessing(nih_metadata_path, nih_image_path, pneumonia_image_path, covid19_image_path, labels):

    (train_set, test_set, valid_set, xray_data) = generate_training_and_testing_set(nih_metadata_path, nih_image_path, pneumonia_image_path, covid19_image_path, labels)

    debug_set(train_set, test_set, valid_set, xray_data)

    (train_datagen, test_datagen) = data_gen_parameters()

    return data_generator(train_datagen, test_datagen, train_set, test_set, valid_set)