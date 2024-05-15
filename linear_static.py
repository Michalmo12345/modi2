import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from initial import u_train, u_test, y_train, y_test, u_data, y_data


# def plot_characteristic():
#     m_matrix = np.vstack([np.ones(len(u_data)), u_data]).T
#     w = np.linalg.lstsq(m_matrix, y_data, rcond=None)[0]
#     y_mod = w[0] + w[1] * u_data
#     plt.figure(figsize=(10, 6))
#     plt.title("Model statyczny liniowy na tle danych trenujących")
#     plt.scatter(u_train, y_mod)
#     plt.scatter(u_train, y_train)
#     plt.show()

def linear():
    m_matrix = np.vstack([np.ones(len(u_train)), u_train]).T
    w = np.linalg.lstsq(m_matrix, y_train, rcond=None)[0]
    y_mod = w[0] + w[1] * u_train
    epsilon = sum((y_mod - y_train) ** 2)
    print(epsilon)


    y_mod_test = w[0] + w[1] * u_test
    epsilon_test = sum((y_mod_test - y_test) ** 2)
    print(epsilon_test)


    plt.figure(figsize=(10, 6))
    plt.title("Model statyczny liniowy na tle danych uczących")
    plt.scatter(u_train, y_mod, color='red', label='Model')
    plt.scatter(u_train, y_train, color='blue', label='Dane uczące')
    plt.legend()
    plt.xlabel('u')
    plt.ylabel('y')
    plt.grid(True)
    plt.savefig('plots/linear_static_train.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title("Model statyczny liniowy na tle danych weryfikujących")
    plt.scatter(u_test, y_mod_test, color='red', label='Model')
    plt.scatter(u_test, y_test, color='orange', label='Dane weryfikujące')
    plt.legend()
    plt.xlabel('u')
    plt.ylabel('y')
    plt.grid(True)
    plt.savefig('plots/linear_static_test.png')
    plt.show()


linear()