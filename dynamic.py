import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data_train = pd.read_csv('data/danedynucz18.txt', delimiter=r"\s+", header=None, names=['U', 'Y'])
data_test = pd.read_csv('data/danedynwer18.txt', delimiter=r"\s+", header=None, names=['U', 'Y'])


u_data_train = data_train['U']
y_data_train = data_train['Y']
u_data_test = data_train['U']
y_data_test = data_train['Y']
u_train = np.array(u_data_train)
u_test = np.array(u_data_test)
y_train = np.array(y_data_train)
y_test = np.array(y_data_test)

def plot_train_split():
    plt.figure(figsize=(8, 6))
    plt.plot(u_train,label='u(k)')
    plt.plot(y_train,label='y(k)')  
    plt.title('Dane uczące')
    plt.legend()  
    plt.grid(True)
    plt.show()

def plot_test_split():
    plt.figure(figsize=(8, 6))
    plt.plot(u_test, label='u(k)')
    plt.plot(y_test, label='y(k)')
    plt.title('Dane uczące')
    plt.legend()
    plt.grid(True)
    plt.show()

def linear_dynamic_1_degree():
    m_matrix_train = np.column_stack((u_train[:-1], y_train[:-1]))
    m_matrix_test = np.column_stack((u_test[:-1], y_test[:-1]))
    w_train= np.linalg.lstsq(m_matrix_train, y_train[1:], rcond=None)[0]
    w_test= np.linalg.lstsq(m_matrix_test, y_test[1:], rcond=None)[0]
    y_mod_train, y_mod_rec_train, y_mod_test, y_mod_rec_test = initialize_model_outputs(y_train, y_test, 1)

    for k in range(1, len(y_train)):
        y_mod_train[k] = w_train[0] * u_train[k-1] + w_train[1] * y_train[k-1]
        y_mod_rec_train[k] = w_train[0] * u_train[k-1] + w_train[1] * y_mod_rec_train[k-1]

    for k in range(1, len(y_test)):
        y_mod_test[k] = w_test[0] * u_test[k-1] + w_test[1] * y_test[k-1]
        y_mod_rec_test[k] = w_test[0] * u_test[k-1] + w_test[1] * y_mod_rec_test[k-1]

    plot_split(y_mod_train,y_mod_test, y_mod_rec_train, y_mod_rec_test, 'Model dynamiczny liniowy 1 rzędu')

    epsilon_train_1 = np.sum((y_mod_train - y_train)**2)
    epsilon_test_1 = np.sum((y_mod_test - y_test)**2)
    epsilon_train_rec_1 = np.sum((y_mod_rec_train - y_train)**2)
    epsilon_test_rec_1 = np.sum((y_mod_rec_test - y_test)**2)

    return epsilon_train_1, epsilon_test_1, epsilon_train_rec_1, epsilon_test_rec_1

def linear_dynamic_2_degree():
    m_matrix_train = np.column_stack((u_train[1:-1],  u_train[:-2], y_train[1:-1],y_train[:-2]))
    w_train = np.linalg.lstsq(m_matrix_train, y_train[2:], rcond=None)[0]
    m_matrix_test = np.column_stack((u_test[1:-1],  u_test[:-2], y_test[1:-1],y_test[:-2]))
    w_test  = np.linalg.lstsq(m_matrix_test, y_test[2:], rcond=None)[0]
    y_mod_train, y_mod_rec_train, y_mod_test, y_mod_rec_test = initialize_model_outputs(y_train, y_test, 2)

    for k in range(2, len(y_train)):
        y_mod_train[k] = w_train[0] * u_train[k-1]  + w_train[1] * u_train[k-2] + w_train[2] * y_train[k-1] + w_train[3] * y_train[k-2]
        y_mod_rec_train[k] = w_train[0] * u_train[k-1] + w_train[1] * u_train[k-2] + w_train[2] * y_mod_rec_train[k-1] + w_train[3] * y_mod_rec_train[k-2]

    for k in range(2, len(y_test)):
        y_mod_test[k] = w_test[0] * u_test[k-1] + w_test[1] * u_test[k-2] + w_test[2] * y_test[k-1] + w_test[3] * y_test[k-2]
        y_mod_rec_test[k] = w_test[0] * u_test[k-1] + w_test[1] * u_test[k-1] + w_test[2] * y_mod_rec_test[k-2] + w_test[3] * y_mod_rec_test[k-2]

    plot_split(y_mod_train,y_mod_test, y_mod_rec_train, y_mod_rec_test, 'Model dynamiczny liniowy 2 rzędu')

    epsilon_train_2 = np.sum((y_mod_train - y_train)**2)
    epsilon_test_2 = np.sum((y_mod_test - y_test)**2)
    epsilon_train_rec_2 = np.sum((y_mod_rec_train - y_train)**2)
    epsilon_test_rec_2 = np.sum((y_mod_rec_test - y_test)**2)

    return epsilon_train_2, epsilon_test_2, epsilon_train_rec_2, epsilon_test_rec_2

def linear_dynamic_3_degree():
    m_matrix_train = np.column_stack((u_train[2:-1],u_train[1:-2], u_train[:-3], y_train[2:-1], y_train[1:-2],  y_train[:-3]))
    m_matrix_test = np.column_stack((u_test[2:-1],u_test[1:-2], u_test[:-3], y_test[2:-1], y_test[1:-2],  y_test[:-3]))
    w_train= np.linalg.lstsq(m_matrix_train, y_train[3:], rcond=None)[0]
    w_test= np.linalg.lstsq(m_matrix_test, y_test[3:], rcond=None)[0]
    y_mod_train, y_mod_rec_train, y_mod_test, y_mod_rec_test = initialize_model_outputs(y_train, y_test, 3)

    for k in range(3, len(y_train)):
        y_mod_train[k] = w_train[0] * u_train[k-1] + w_train[1] * u_train[k-2] + w_train[2] * u_train[k-3] + w_train[3] * y_train[k-1] + w_train[4] * y_train[k-2] + w_train[5] * y_train[k-3]
        y_mod_rec_train[k] = w_train[0] * u_train[k-1] + w_train[1] * u_train[k-2] + w_train[2] * u_train[k-3] + w_train[3] * y_mod_rec_train[k-1] + w_train[4] * y_mod_rec_train[k-2] + w_train[5] * y_mod_rec_train[k-3]

    for k in range(3, len(y_test)):
        y_mod_test[k] = w_test[0] * u_test[k-1] + w_test[1] * u_test[k-2] + w_test[2] * u_test[k-3] + w_test[3] * y_test[k-1] + w_test[4] * y_test[k-2] + w_test[5] * y_test[k-3]
        y_mod_rec_test[k] = w_test[0] * u_test[k-1] + w_test[1] * u_test[k-2] + w_test[2] * u_test[k-3] + w_test[3] * y_mod_rec_test[k-1] + w_test[4] * y_mod_rec_test[k-2] + w_test[5] * y_mod_rec_test[k-3]

    plot_split(y_mod_train,y_mod_test, y_mod_rec_train, y_mod_rec_test, 'Model dynamiczny liniowy 3 rzędu')

    epsilon_train_3 = np.sum((y_mod_train - y_train)**2)
    epsilon_test_3 = np.sum((y_mod_test - y_test)**2)
    epsilon_train_rec_3 = np.sum((y_mod_rec_train - y_train)**2)
    epsilon_test_rec_3 = np.sum((y_mod_rec_test - y_test)**2)

    return epsilon_train_3, epsilon_test_3, epsilon_train_rec_3, epsilon_test_rec_3

def initialize_model_outputs(y_train, y_test, degree):
    y_mod_train = np.zeros_like(y_train)
    y_mod_test = np.zeros_like(y_test)
    y_mod_rec_train = np.zeros_like(y_train)
    y_mod_rec_test = np.zeros_like(y_test)

    for i in range(degree):
        y_mod_train[i] = y_train[i]
        y_mod_test[i] = y_test[i]
        y_mod_rec_train[i] = y_train[i]
        y_mod_rec_test[i] = y_test[i]

    return y_mod_train, y_mod_rec_train, y_mod_test, y_mod_rec_test


def calculate_dynamic_errors():
    results = {
        'Degree': [], 
        'Epsilon_train': [], 
        'Epsilon_test': [],
        'Epsilon_train_rec': [],
        'Epsilon_test_rec': []}
    for degree in range(1, 4):
        results['Degree'].append(degree)
        if degree == 1:
            epsilon_train, epsilon_test, epsilon_train_rec, epsilon_test_rec = linear_dynamic_1_degree()
        elif degree == 2:
            epsilon_train, epsilon_test, epsilon_train_rec, epsilon_test_rec = linear_dynamic_2_degree()
        else:
            epsilon_train, epsilon_test, epsilon_train_rec, epsilon_test_rec = linear_dynamic_3_degree()
        results['Epsilon_train'].append(epsilon_train)
        results['Epsilon_test'].append(epsilon_test)
        results['Epsilon_train_rec'].append(epsilon_train_rec)
        results['Epsilon_test_rec'].append(epsilon_test_rec)
    df_results = pd.DataFrame(results)
    return df_results

def plot_split(y_mod_train,y_mod_test, y_mod_rec_train, y_mod_rec_test, title):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.plot(y_mod_train, label='Wyjście modelu')
    plt.plot(y_train, label='Dane trenujące')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.plot(y_mod_test, label='Wyjście modelu ')
    plt.plot(y_test, label='Dane testowe')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.plot(y_mod_rec_train, label='Wyjście modelu z rekurencja')
    plt.plot(y_train, label='Dane trenujące')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.plot(y_mod_rec_test, label='Wyjście modelu z rekurencja')
    plt.plot(y_test, label='Dane testowe')
    plt.legend()
    plt.grid(True)
    plt.show()



def create_m_matrix(n, degree, u_data, y_data):
    num_samples = len(u_data) - n
    num_features = degree * n * 2
    feature_matrix = np.zeros((num_samples, num_features))
    for i in range(num_samples):
        feature_row = []
        k = i + n

        for lag in range(1, n + 1):
            for power in range(1, degree + 1):
                feature_row.append(u_data[k - lag] ** power)

        for lag in range(1, n + 1):
            for power in range(1, degree + 1):
                feature_row.append(y_data[k - lag] ** power)

        feature_matrix[i, :] = feature_row

    return feature_matrix


def non_liner_dynamic(n_row ,degree):
    m_matrix_train = create_m_matrix(n_row, degree, u_train, y_train)
    m_matrix_test = create_m_matrix(n_row, degree, u_test, y_test)
    w_train = np.linalg.lstsq(m_matrix_train, y_train[n_row:], rcond=None)[0]
    w_test = np.linalg.lstsq(m_matrix_test, y_test[n_row:], rcond=None)[0]
    y_mod_train, y_mod_rec_train, y_mod_test, y_mod_rec_test = initialize_model_outputs(y_train, y_test, n_row)
    # y_mod_train[n_row:] = np.dot(m_matrix_train, w_train)
    # y_mod_test[n_row:] = np.dot(m_matrix_test, w_test)
    
    for k in range(n_row, len(y_train)):
        for i in range(1, degree + 1):
            y_mod_train[k] = w_train[i - 1] * u_train[k - 1] ** i + w_train[degree + i - 1] * y_train[k - 1] ** i
            y_mod_rec_train[k] = w_train[i - 1] * u_train[k - 1] ** i + w_train[degree + i - 1] * y_mod_rec_train[k - 1] ** i

    for k in range(n_row, len(y_test)):
        for i in range(1, degree + 1):
            y_mod_test[k] = w_test[i - 1] * u_test[k - 1] ** i + w_test[degree + i - 1] * y_test[k - 1] ** i
            y_mod_rec_test[k] = w_test[i - 1] * u_test[k - 1] ** i + w_test[degree + i - 1] * y_mod_rec_test[k - 1] ** i

    return y_mod_train, y_mod_test, y_mod_rec_train, y_mod_rec_test


def calculate_non_linear_errors(n_row, degree):
    results = {
        'Degree': [], 
        'Epsilon_train': [], 
        'Epsilon_test': [],
        'Epsilon_train_rec': [],
        'Epsilon_test_rec': []}
    for degree in range(1, degree + 1):
        results['Degree'].append(degree)
        y_mod_train, y_mod_test, y_mod_rec_train, y_mod_rec_test = non_liner_dynamic(n_row, degree)
        epsilon_train = np.sum((y_mod_train - y_train) ** 2)
        epsilon_test = np.sum((y_mod_test - y_test) ** 2)
        epsilon_train_rec = np.sum((y_mod_rec_train - y_train) ** 2)
        epsilon_test_rec = np.sum((y_mod_rec_test - y_test) ** 2)
        results['Epsilon_train'].append(epsilon_train)
        results['Epsilon_test'].append(epsilon_test)
        results['Epsilon_train_rec'].append(epsilon_train_rec)
        results['Epsilon_test_rec'].append(epsilon_test_rec)
    df_results = pd.DataFrame(results)
    return df_results

# linear_dynamic_1_degree()
# linear_dynamic_2_degree()
# linear_dynamic_3_degree()
# df_results = calculate_dynamic_errors()
# print(df_results)
non_linear_eps = calculate_non_linear_errors(2, 3)

print(non_linear_eps)

