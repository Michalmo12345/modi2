import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from modi1 import u_train, u_test, y_train, y_test




def polynomial_train(degree):
    m_matrix = np.vstack([u_train**i for i in range(degree + 1)]).T
    w = np.linalg.lstsq(m_matrix, y_train, rcond=None)[0]

    y_mod = sum(w[i] * u_train**i for i in range(degree + 1))

    epsilon = np.sum((y_mod - y_train)**2)
    print(epsilon)

    plt.figure(figsize=(10, 6))
    plt.title("Model statyczny liniowy na tle danych trenujących")
    plt.scatter(u_train, y_mod)
    plt.scatter(u_train, y_train)
    plt.show()
    return epsilon

def polynomial_test(degree):
    m_matrix = np.vstack([u_test**i for i in range(degree + 1)]).T
    w = np.linalg.lstsq(m_matrix, y_test, rcond=None)[0]

    y_mod = sum(w[i] * u_test**i for i in range(degree + 1))

    epsilon = np.sum((y_mod - y_test)**2)

    plt.figure(figsize=(10, 6))
    plt.title("Model statyczny liniowy na tle danych testowych")
    plt.scatter(u_test, y_mod)
    plt.scatter(u_test, y_test)
    plt.show()
    return epsilon


def calculate_polynomial_errors(max_degree):
    results = {'Degree': [], 'Epsilon_train': [], 'Epsilon_test': []}
    for degree in range(1, max_degree+1):
        results['Degree'].append(degree)
        epsilon_train  = polynomial_train(degree)
        epsilon_test = polynomial_test(degree)
        results['Epsilon_train'].append(epsilon_train)
        results['Epsilon_test'].append(epsilon_test)
    df_results = pd.DataFrame(results)
    return df_results


polynomial_train(5)
print(calculate_polynomial_errors(8))