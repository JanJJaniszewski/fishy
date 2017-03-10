import glob
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def get_xy(folders_paths=['../Data/Kaggle/train/*', '../Data/Imagenet/*'],
           test_size=0.0, img_size=(224, 224), seed=7):
    """Retrieves all data from a folders_path and gives back
    x_train, x_test, y_train, y_test
    folders_path: The folder to the data (standard: '../Data/Kaggle/train/*')
    img_size: size of the image (standard: (224, 224))
    split: Where to make the split between training and test data
    """
    # creating a dataframe with the columns below
    image_dict = {}

    # going through all folders and saving file names in dataframe so that order is always correct
    for folders_path in folders_paths:
        for foldername in glob.glob(folders_path):
            for image_name in glob.glob(foldername + '/*.jpg'):
                # getting image name and fish type and putting it in a dictionary
                fish_type = foldername.split('/')[-1]
                image_dict[image_name] = fish_type
            
        assert image_dict, 'Files not found in {}'.format(folders_path)

    # image_df: image_name | fish_type
    image_df = pd.DataFrame({'image': list(image_dict.keys()), 'fish_type': list(image_dict.values())}).sample(frac=1)   
    
    # dummy_df: dummy version of image_df
    dummy_df = pd.get_dummies(image_df, columns=['fish_type']) # dummify data
    
    # Create list with images as arrays
    images = np.array([cv2.imread(img) for img in dummy_df.image])   
    images = [cv2.resize(img, (224, 224), cv2.INTER_LINEAR) for img in images]

    # X and y value and labels
    x = images
    y = dummy_df.iloc[:,1:9].as_matrix()

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
    
    x_train, x_test, y_train, y_test = np.asarray(x_train), np.asarray(x_test), np.asarray(y_train), np.asarray(y_test)
    
    
    # Return stuff
    return x_train, y_train, x_test, y_test, image_df, dummy_df


def get_final_test(files_path='../Data/Kaggle/test1/*.jpg', img_size=(224, 224)):
    """Gets final test data
    """
    # going through all folders and saving file names in list so that order is always correct
    x_test_names = glob.glob(files_path)
    x_test = np.asarray([cv2.resize(cv2.imread(name), (224, 224), cv2.INTER_LINEAR) for name in x_test_names])
    
    # Return stuff
    return x_test_names, x_test