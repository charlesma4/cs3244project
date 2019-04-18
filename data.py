import os
import sys
import glob
import pandas as pd
import numpy as np
import random
import threading

from preprocess import preprocess_csv_data, process_image
from tools import rescale_list, get_frames
from sklearn.model_selection import train_test_split


class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class Data:
    def __init__(self, num_frames=48, image_shape=(160, 120)):
        self.num_frames = num_frames
        self.image_shape = image_shape
        self.classes = {
            'fall': 1,
            'adl': 0
        }

    def load_all_sequence_data(self, split=True):
        # Loads into memory all of the sequence data

        data_path = os.path.join('data', 'sequences')
        sequences = os.listdir(data_path)
        print('Extracting sequences...')

        X, y = [], []
        for seq in sequences:
            
            label = self.classes[seq.split('-')[0]]
            frames = get_frames(os.path.join(data_path, seq))
            frames = rescale_list(frames, self.num_frames, images=True)
            sequence = self.build_sequence(frames)

            X.append(sequence)
            y.append(label)

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        print('---- X shape: {}\n---- y shape: {}'.format(X.shape, y.shape))

        if split:
            return train_test_split(X, y, test_size=0.3, random_state=1998)
        else:
            return X, y
    
    def load_image_train_split(self):
        data_path = os.path.join('data', 'sequences')
        sequences = os.listdir(data_path)
         print('Extracting sequences...')

        X, y = [], []
        for seq in sequences:
            label = self.classes[seq.split('-')[0]]
            frames = get_frames(os.path.join(data_path, seq))
            frames = rescale_list(frames, self.num_frames, images=True)

            X.append(frames)
            y.append(label)

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        print('---- X shape: {}\n---- y shape: {}'.format(X.shape, y.shape))
        self.num_samples = len(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=1998)

    def load_extracted_data(self, split=True):
        data_folder = os.path.join('data', 'extracted')
        if not os.path.isfile(os.path.join(data_folder, 'data.npy')):
            preprocess_csv_data(self.num_frames)

        X = np.load(os.path.join(data_folder, 'data.npy'))

        # If new input shape does not match the cached one, resave...
        if X.shape[1] != self.num_frames:
            preprocess_csv_data(self.num_frames)
            X = np.load(os.path.join(data_folder, 'data.npy'))

        y = np.load(os.path.join(data_folder, 'labels.npy')).reshape(-1, 1)
        print('---- X shape: {}\n---- y shape: {}'.format(X.shape, y.shape))

        if split:
            return train_test_split(X, y, test_size=0.3, random_state=1998)
        else:
            return X, y

    def build_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [process_image(x, self.image_shape) for x in frames]

    @threadsafe_generator
    def frame_generator(self, batch_size, train_test):
        data = self.X_train if train_test == 'train' else self.X_test
        data_path = os.path.join('data', 'sequences')

        print("Creating %s generator with %d samples." % (train_test, len(data)))

        while 1:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                sequence = None

                # Get a random sample.
                sample = random.choice(data)
                print(sample)

                label = self.classes[sample[0].split('-')[0]]
                frames = get_frames(os.path.join(data_path, sample))
                frames = rescale_list(frames, self.num_frames, images=True)
                sequence = self.build_sequence(frames)

                X.append(sequence)
                y.append(label)

            yield np.array(X), np.array(y).reshape(-1, 1)