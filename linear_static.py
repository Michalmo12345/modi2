import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from modi1 import u_train, u_test, y_train, y_test

def linear_train():
    m_matrix = np.vstack([np.ones(len(u_train)), u_train]).T
    w = np.linalg.lstsq(m_matrix, y_train, rcond=None)[0]

    y_mod = w[0] + w[1] * u_train

    epsilon = 0
    for i in range(0, len(y_train)):
        epsilon = epsilon + (y_mod[i] - y_train[i])**2
    print(epsilon)

    plt.figure(figsize=(10, 6))
    plt.title("Model statyczny liniowy na tle danych trenujących")
    plt.scatter(u_train, y_mod)
    plt.scatter(u_train, y_train)
    plt.show()

def linear_test():
    m_matrix = np.vstack([np.ones(len(u_test)), u_test]).T
    w = np.linalg.lstsq(m_matrix, y_test, rcond=None)[0]

    y_mod = w[0] + w[1] * u_test

    epsilon = 0
    for i in range(0, len(y_test)):
        epsilon = epsilon + (y_mod[i] - y_test[i])**2
    print(epsilon)

    plt.figure(figsize=(10, 6))
    plt.title("Model statyczny liniowy na tle danych testowych")
    plt.scatter(u_test, y_mod)
    plt.scatter(u_test, y_test)
    plt.show()

linear_train()
linear_test()