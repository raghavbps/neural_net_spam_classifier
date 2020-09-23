import argparse

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras import regularizers
from neural_net_spam_classifier.container.source_dir.preprocess import create_training_data
import os


def load_neural_network_model():
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(512,)),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    print(model.summary())
    return model


def fit_neural_network_model(x, y, model, model_dir):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, shuffle=True)
    model.fit(x_train, y_train, epochs=20, batch_size=1)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    model.save(os.path.join(model_dir, 'nn_model.h5'))
    print('Test accuracy:', test_acc)


if __name__ == '__main__':
    print('Started Training')
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data/training')
    args_parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
    args_parser.add_argument('--learning-rate', type=float, default=0.01, help='Initial learning rate.')
    args_parser.add_argument('--epochs', type=int, default=50, help='The number of steps to use for training.')
    args_parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training.')
    args_parser.add_argument('--optimizer', type=str.lower, default='adam')
    args = args_parser.parse_args()
    print(args)
    x, y = create_training_data(args.data_dir)
    fit_neural_network_model(x, y, model=load_neural_network_model(), model_dir=args.model_dir)
