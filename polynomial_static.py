import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from initial import u_train, u_test, y_train, y_test


def polynomial_train(degree):
    m_matrix = np.vstack([u_train**i for i in range(degree + 1)]).T
    w = np.linalg.lstsq(m_matrix, y_train, rcond=None)[0]

    y_mod_train = sum(w[i] * u_train**i for i in range(degree + 1))
    y_mod_test = sum(w[i] * u_test**i for i in range(degree + 1))

    epsilon = np.sum((y_mod_train - y_train)**2)
    epsilon_test = np.sum((y_mod_test - y_test)**2)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    axs[0].scatter(u_train, y_mod_train, color='red', label='Model')
    axs[0].scatter(u_train, y_train, color='blue', label='Dane uczące')
    axs[0].set_title("Model statyczny nieliniowy na tle danych uczących, wielomian stopnia " + str(degree))
    axs[0].set_xlabel('u')
    axs[0].set_ylabel('y')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].scatter(u_test, y_mod_test, color='red', label='Model')
    axs[1].scatter(u_test, y_test, color='orange', label='Dane weryfikujące')
    axs[1].set_title("Model statyczny nieliniowy na tle danych weryfikujących, wielomian stopnia " + str(degree))
    axs[1].set_xlabel('u')
    axs[1].set_ylabel('y')
    axs[1].legend()
    axs[1].grid(True)


    # print(f"Training error (epsilon): {epsilon}")
    # print(f"Testing error (epsilon_test): {epsilon_test}")

    plt.tight_layout()
    # plt.savefig(f'plots/polynomial_static_degree_{degree}.png')
    # plt.show()
    return epsilon, epsilon_test


def calculate_polynomial_errors(max_degree):
    results = {'Degree': [], 'Epsilon_train': [], 'Epsilon_test': []}
    for degree in range(1, max_degree+1):
        results['Degree'].append(degree)
        epsilon_train, epsilon_test  = polynomial_train(degree)
        results['Epsilon_train'].append(epsilon_train)
        results['Epsilon_test'].append(epsilon_test)
    df_results = pd.DataFrame(results)
    df_results = df_results.round(3)
    df_results.to_csv('results/polynomial_static_errors.csv', index=False)
    return df_results


# polynomial_train(15)
print(calculate_polynomial_errors(15))