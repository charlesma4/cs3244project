import os
import pathlib
import pandas as pd
import numpy as np
from tools import rescale_list
from keras.preprocessing.image import img_to_array, load_img

def process_image(image, target_shape):
    # Load the image.
    h, w = target_shape
    image = load_img(image, target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    x = (img_arr / 255.).astype(np.float32)

    return x

def preprocess_csv_data(num_frames):
    raw_data_fall = os.path.join("data","raw","urfall-cam0-falls.csv")
    raw_data_adls = os.path.join("data","raw","urfall-cam0-adls.csv")
    
    X, y = [], []
    curr_sequence, curr_index = None, -1

    # First load fall data
    df = pd.read_csv(raw_data_fall)
    for _, row in df.iterrows():
        if row['name'] != curr_sequence:
            curr_sequence = row['name']
            curr_index += 1
            X.append([])
            y.append(1)
        X[curr_index].append(row.tolist()[3:])
    
    # Then read adl data
    df = pd.read_csv(raw_data_adls)
    for _, row in df.iterrows():
        if row['name'] != curr_sequence:
            curr_sequence = row['name']
            curr_index += 1
            X.append([])
            y.append(0)
        X[curr_index].append(row.tolist()[3:])

    # Rescale each sequence into num_frames
    for i in range(len(X)):
        X[i] = rescale_list(X[i], num_frames)
    
    # sanity check
    assert len(X) == len(y)
    X = np.array(X)
    y = np.array(y)
    print('---- Feature shape: {}\n---- Label shape: {}'.format(X.shape, y.shape))
    
    data_folder = os.path.join('data', 'extracted')
    pathlib.Path(data_folder).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(data_folder, 'data'), X)
    np.save(os.path.join(data_folder, 'labels'), y)