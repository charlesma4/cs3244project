import os
import sys
import glob
import pandas as pd
import numpy as np

from preprocess import preprocess_csv_data, process_image
from tools import rescale_list, get_frames
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, num_frames=48, image_shape=(224, 224, 3)):
        self.num_frames = num_frames
        self.classes = {
            'fall': 1,
            'adl': 0
        }
    
    def load_sequence_data(self):
        # Loads into memory all of the sequence data

        data_path = os.path.join('data', 'sequences')
        sequences = os.listdir(data_path)

        X, y = [], []
        for seq in sequences:
            label = self.classes[seq.split('-')[0]]
            frames = get_frames(os.path.join(data_path, seq))
            frames = rescale_list(frames, self.num_frames)
            sequence = self.build_sequence(self.num_frames)

            X.append(sequence)
            y.append(label)

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        return train_test_split(X, y, test_size=0.4, random_state=1998)
    
    def load_extracted_data(self):
        data_folder = os.path.join('data', 'extracted')
        if not os.path.isfile(os.path.join(data_folder, 'data.npy')):
            preprocess_csv_data(self.num_frames)
        X = np.load(os.path.join(data_folder, 'data.npy'))
        if X.shape[1] != self.num_frames:
            preprocess_csv_data(self.num_frames)
            X = np.load(os.path.join(data_folder, 'data.npy'))
        y = np.load(os.path.join(data_folder, 'labels.npy')).reshape(-1, 1)
        return train_test_split(X, y, test_size=0.4, random_state=1998)

    def build_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [process_image(x, self.num_frames) for x in frames]