import time
import os.path
import sys
import pathlib

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from model import Model
from data import Data


def train(model_name, num_frames=48, num_features=4, saved_model=None,
          image_shape=None, num_samples=70, save_trained_model=True, fold_validate=False,
          load_to_memory=False, batch_size=1, nb_epoch=100, drop_out=0.3):

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model_name))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model_name + '-' + 'training-' +
                                        str(timestamp) + '.log'))

    # Get the data and process it.
    data = Data(num_frames=num_frames, image_shape=image_shape)

    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    # steps_per_epoch = (num_samples * 0.7) // batch_size

    # Get the model.
    rm = Model(model_name, num_frames=num_frames,
               saved_model=saved_model, image_shape=image_shape)

    if fold_validate:
        if model_name == 'lstm':
            X, y = data.load_extracted_data(split=False) 
        else:
            X,y = data.load_sequence_data(split=False)

        kf = KFold(n_splits=10)
        tn, fp, fn, tp = 0, 0, 0, 0
        count = 1
        for train, test in kf.split(X, y):
            rm.model.fit(
                X[train],
                y[train],
                batch_size=batch_size,
                validation_data=(X[test], y[test]),
                verbose=0,
                callbacks=[tb, early_stopper, csv_logger],
                epochs=nb_epoch)

            test_loss, test_acc = rm.model.evaluate(X[test], y[test])
            print('Fold {}\n---- Test Accuracy: {}\n---- Test Loss: {}\n'.format(count, test_acc, test_loss))

            y_pred = rm.model.predict(X[test])
            y_pred = (y_pred > 0.5)

            # tn, fp, fn, tp
            tn1,fp1,fn1,tp1 = confusion_matrix(y[test], y_pred)
            tn += tn1
            fp += fp1
            fn += fn1
            tp += tp1
            count+=1
        print('tn: {}, fp: {}, fn: {}, tp: {}'.format(tn, fp, fn, tp))
    else:
        if model_name == 'lstm':
            X_train, X_test, y_train, y_test = data.load_extracted_data()
        elif model_name == 'lrcn':
            X_train, X_test, y_train, y_test = data.load_sequence_data()
        rm.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger],
            epochs=nb_epoch)

        test_loss, test_acc = rm.model.evaluate(X_test, y_test)
        print('Test Accuracy: {}\nTest Loss: {}'.format(test_acc, test_loss))

        y_pred = rm.model.predict(X_test)
        y_pred = (y_pred > 0.5)

        # tn, fp, fn, tp
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

    if save_trained_model:
        save_model_name = 'model-{}-{}.h5'.format(model_name, test_acc)
        if not os.path.isdir(os.path.join('data', 'trained')):
            pathlib.Path(os.path.join('data', 'trained')).mkdir(
                parents=True, exist_ok=True)
        if not os.path.isfile(os.path.join('data', 'trained', save_model_name)):
            rm.model.save(os.path.join('data', 'trained', save_model_name))


def main():
    model_name = 'lstm'
    saved_model = None  # None or weights file
    save_trained_model = True
    batch_size = 10
    nb_epoch = 100
    image_shape = (80, 80, 3)
    fold_validate = True

    if len(sys.argv) > 1:
        num_frames = int(sys.argv[1])
    else:
        num_frames = 120
    num_features = 4

    train(model_name, num_frames=num_frames, saved_model=saved_model, image_shape=image_shape, fold_validate=fold_validate,
          batch_size=batch_size, nb_epoch=nb_epoch, num_features=num_features, save_trained_model=save_trained_model)


if __name__ == '__main__':
    main()
