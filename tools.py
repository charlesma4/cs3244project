import os
import sys
import glob
from keras import backend as K

def rescale_list(input_list, size):
    # Either scale list down if input length is less than size or pad with 0s.
    if not input_list:
        print('Input list is empty.')
        sys.exit()

    if len(input_list) > size:
        # Skip frames to get to correct length
        skip = len(input_list) // size
        output = [input_list[i] for i in range(0, len(input_list), skip)]
    else:
        output = input_list
        for _ in range(len(input_list), size):
            output.append([0 for __ in range(len(input_list[0]))])
    print(len(output), len(output[0]))
    return output[:size]

def get_frames(path):
    images = sorted(glob.glob(os.path.join(path, '*jpg')))
    return images

# Custom F1 metric @Paddy and @Kev1n91 on StackOverflow
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))