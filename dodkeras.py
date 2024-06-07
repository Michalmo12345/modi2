from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, SimpleRNN, LSTM, LeakyReLU
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def create_data(N, data_type, recursive):
    train_data = pd.read_csv('data/danedynucz18.txt', delimiter=r"\s+", header=None, names=['U', 'Y'])
    test_data = pd.read_csv('data/danedynwer18.txt', delimiter=r"\s+", header=None, names=['U', 'Y'])
    data = train_data if data_type else test_data
    data = np.array(data)
    input_data = []
    output_data = []
    for i in range(N-1, len(data[:, 0])):
        row = []
        for j in range(1, N):
            row.append(data[i - j, 0])  # U data
            row.append(data[i - j, 1])  # Y data
        input_data.append(row)
        output_data.append(data[i, 1])  # Current Y
    input_data = np.array(input_data)
    if recursive:
        input_data = input_data.reshape((input_data.shape[0], N-1, 2))
    return input_data, np.array(output_data)

def create_model(recursive, k, N):
    model = Sequential()
    input_layer = LSTM(k, input_shape=(N-1, 2), activation='leaky_relu') if recursive else Dense(k, input_dim=2 * (N - 1), activation='leaky_relu')
    model.add(input_layer)
    model.add(Dense(1, activation='leaky_relu'))
    return model


def neural_network(N, k, recursive):
    train_data = pd.read_csv('data/danedynucz18.txt', delimiter=r"\s+", header=None, names=['U', 'Y'])
    test_data = pd.read_csv('data/danedynwer18.txt', delimiter=r"\s+", header=None, names=['U', 'Y'])
    input_data, output_data = create_data(N, True, recursive)
    # Model creation
    model = create_model(recursive, k, N)

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(input_data, output_data, epochs=100)

    input_data_test, output_data_test = create_data(N, False, recursive)

    predictions_train = model.predict(input_data)
    predictions_test = model.predict(input_data_test)

    predictions_train = predictions_train.flatten()
    predictions_test = predictions_test.flatten()

    epsilon_train = round(np.sum((output_data - predictions_train)**2), 3)
    epsilon_test = round(np.sum((output_data_test - predictions_test)**2), 3)

    recursive_str = 'z rekurencją' if recursive else 'bez rekurencji'
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(output_data, label='Dane trenujące')
    plt.plot(predictions_train, label='Model')
    plt.title(f'Model {recursive_str} - liczba neuronów w warstwie:{k}, dane trenujące')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(output_data_test, label='Dane testowe')
    plt.plot(predictions_test, label='Model')
    plt.title(f'Model {recursive_str}- liczba neuronów w warstwie: {k}, dane testowe')
    plt.legend()
    plt.show()

    return epsilon_train, epsilon_test



neural_network(11, 1, recursive=False)
neural_network(11, 1, recursive=True)
neural_network(11, 64, recursive=False)
neural_network(11, 64, recursive=True)

