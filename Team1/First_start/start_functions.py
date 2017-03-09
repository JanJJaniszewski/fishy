from imports import *

def get_xy(folders_path='Data/Kaggle/train/*', split=0.8, img_size=(224, 224)):
    """Retrieves all data from a folders_path and gives back
    x_train, x_test, y_train, y_test
    folders_path: The folder to the data (standard: '../Data/Kaggle/train/*')
    img_size: size of the image (standard: (224, 224))
    split: Where to make the split between training and test data
    """
    # creating a dataframe with the columns below
    image_dict = {}

    # going through all folders and saving file names in dataframe so that order is always correct
    for foldername in glob.glob(folders_path):
        for image_name in glob.glob(foldername + '/*.jpg'):
            # getting image name and fish type and putting it in a dictionary
            fish_type = foldername[-3:]
            image_dict[image_name] = fish_type
            
    assert image_dict, 'Files not found in {}'.format(folders_path)

    # image_df: image_name | fish_type
    image_df = pd.DataFrame(image_dict.items(), columns = ['image_name', 'fish_type']).sample(frac=1)
    
    # dummy_df: dummy version of image_df
    dummy_df = pd.get_dummies(image_df, columns=['fish_type']) # dummify data
    
    # Create list with images as arrays
    images = []
    for filename in dummy_df.image_name:
        img = image.load_img(filename, target_size=img_size)
        x = image.img_to_array(img)    
        images.append(x)

    # X and y value and labels
    x = np.asarray(images)
    y = dummy_df.iloc[:,1:9].as_matrix()
    labels = dummy_df.image_name.apply(lambda path: path[-13:])

    # table, just to be sure that everything goes well
    dummy_df['image_bytes'] = images

    # Split dataset
    split_no = int(len(x) * split)

    x_train = x[:split_no]
    y_train = y[:split_no]
    labels_train = labels[:split_no]

    x_test = x[split_no:]
    y_test = y[split_no:]
    labels_test = labels[split_no:]
    
    # Return stuff
    return x_train, y_train, labels_train, x_test, y_test, labels_test, image_df, dummy_df