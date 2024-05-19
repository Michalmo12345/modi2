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
    plt.savefig('plots/dynamic_train.png')
    plt.show()

def plot_test_split():
    plt.figure(figsize=(8, 6))
    plt.plot(u_test, label='u(k)')
    plt.plot(y_test, label='y(k)')
    plt.title('Dane weryfikujące')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/dynamic_test.png')
    plt.show()

def linear_dynamic_1_degree():
    m_matrix = np.column_stack((u_train[:-1], y_train[:-1]))
    w= np.linalg.lstsq(m_matrix, y_train[1:], rcond=None)[0]
    y_mod_train, y_mod_rec_train, y_mod_test, y_mod_rec_test = initialize_model_outputs(y_train, y_test, 1)

    for k in range(1, len(y_train)):
        y_mod_train[k] = w[0] * u_train[k-1] + w[1] * y_train[k-1]
        y_mod_rec_train[k] = w[0] * u_train[k-1] + w[1] * y_mod_rec_train[k-1]

    for k in range(1, len(y_test)):
        y_mod_test[k] = w[0] * u_test[k-1] + w[1] * y_test[k-1]
        y_mod_rec_test[k] = w[0] * u_test[k-1] + w[1] * y_mod_rec_test[k-1]

    plot_split(y_mod_train,y_mod_test, y_mod_rec_train, y_mod_rec_test, 'Model dynamiczny liniowy 1 rzędu')

    epsilon_train_1 = np.sum((y_mod_train - y_train)**2)
    epsilon_test_1 = np.sum((y_mod_test - y_test)**2)
    epsilon_train_rec_1 = np.sum((y_mod_rec_train - y_train)**2)
    epsilon_test_rec_1 = np.sum((y_mod_rec_test - y_test)**2)

    return epsilon_train_1, epsilon_test_1, epsilon_train_rec_1, epsilon_test_rec_1

def linear_dynamic_2_degree():
    m_matrix_train = np.column_stack((u_train[1:-1],  u_train[:-2], y_train[1:-1],y_train[:-2]))
    w = np.linalg.lstsq(m_matrix_train, y_train[2:], rcond=None)[0]
    y_mod_train, y_mod_rec_train, y_mod_test, y_mod_rec_test = initialize_model_outputs(y_train, y_test, 2)

    for k in range(2, len(y_train)):
        y_mod_train[k] = w[0] * u_train[k-1]  + w[1] * u_train[k-2] + w[2] * y_train[k-1] + w[3] * y_train[k-2]
        y_mod_rec_train[k] = w[0] * u_train[k-1] + w[1] * u_train[k-2] + w[2] * y_mod_rec_train[k-1] + w[3] * y_mod_rec_train[k-2]

    for k in range(2, len(y_test)):
        y_mod_test[k] = w[0] * u_test[k-1] + w[1] * u_test[k-2] + w[2] * y_test[k-1] + w[3] * y_test[k-2]
        y_mod_rec_test[k] = w[0] * u_test[k-1] + w[1] * u_test[k-1] + w[2] * y_mod_rec_test[k-2] + w[3] * y_mod_rec_test[k-2]

    plot_split(y_mod_train,y_mod_test, y_mod_rec_train, y_mod_rec_test, 'Model dynamiczny liniowy 2 rzędu')

    epsilon_train_2 = np.sum((y_mod_train - y_train)**2)
    epsilon_test_2 = np.sum((y_mod_test - y_test)**2)
    epsilon_train_rec_2 = np.sum((y_mod_rec_train - y_train)**2)
    epsilon_test_rec_2 = np.sum((y_mod_rec_test - y_test)**2)

    return epsilon_train_2, epsilon_test_2, epsilon_train_rec_2, epsilon_test_rec_2

def linear_dynamic_3_degree():
    m_matrix_train = np.column_stack((u_train[2:-1],u_train[1:-2], u_train[:-3], y_train[2:-1], y_train[1:-2],  y_train[:-3]))
    w= np.linalg.lstsq(m_matrix_train, y_train[3:], rcond=None)[0]
    y_mod_train, y_mod_rec_train, y_mod_test, y_mod_rec_test = initialize_model_outputs(y_train, y_test, 3)

    for k in range(3, len(y_train)):
        y_mod_train[k] = w[0] * u_train[k-1] + w[1] * u_train[k-2] + w[2] * u_train[k-3] + w[3] * y_train[k-1] + w[4] * y_train[k-2] + w[5] * y_train[k-3]
        y_mod_rec_train[k] = w[0] * u_train[k-1] + w[1] * u_train[k-2] + w[2] * u_train[k-3] + w[3] * y_mod_rec_train[k-1] + w[4] * y_mod_rec_train[k-2] + w[5] * y_mod_rec_train[k-3]

    for k in range(3, len(y_test)):
        y_mod_test[k] = w[0] * u_test[k-1] + w[1] * u_test[k-2] + w[2] * u_test[k-3] + w[3] * y_test[k-1] + w[4] * y_test[k-2] + w[5] * y_test[k-3]
        y_mod_rec_test[k] = w[0] * u_test[k-1] + w[1] * u_test[k-2] + w[2] * u_test[k-3] + w[3] * y_mod_rec_test[k-1] + w[4] * y_mod_rec_test[k-2] + w[5] * y_mod_rec_test[k-3]

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

    # plot_split(y_mod_train, y_mod_test, y_mod_rec_train, y_mod_rec_test)
    return y_mod_train, y_mod_rec_train, y_mod_test, y_mod_rec_test


# def calculate_dynamic_errors():
#     results = {
#         'Degree': [], 
#         'Epsilon_train': [], 
#         'Epsilon_test': [],
#         'Epsilon_train_rec': [],
#         'Epsilon_test_rec': []}
#     for degree in range(1, 4):
#         results['Degree'].append(degree)
#         if degree == 1:
#             epsilon_train, epsilon_test, epsilon_train_rec, epsilon_test_rec = linear_dynamic_1_degree()
#         elif degree == 2:
#             epsilon_train, epsilon_test, epsilon_train_rec, epsilon_test_rec = linear_dynamic_2_degree()
#         else:
#             epsilon_train, epsilon_test, epsilon_train_rec, epsilon_test_rec = linear_dynamic_3_degree()
#         results['Epsilon_train'].append(epsilon_train)
#         results['Epsilon_test'].append(epsilon_test)
#         results['Epsilon_train_rec'].append(epsilon_train_rec)
#         results['Epsilon_test_rec'].append(epsilon_test_rec)
#     df_results = pd.DataFrame(results)
#     return df_results
def calculate_dynamic_errors():
    results = {
        'Rodzaj danych:': [], 
        'Wyznaczenie wartości:': [],
        'N=1': [], 
        'N=2': [], 
        'N=3': []
    }
    
    degrees = [1, 2, 3]
    data_types = ['Dane uczące', 'Dane uczące', 'Dane weryfikujące', 'Dane weryfikujące']
    calc_types = ['Bez rekurecji', 'Z rekurecją', 'Bez rekurecji', 'Z rekurecją']
    
    for i, (data_type, calc_type) in enumerate(zip(data_types, calc_types)):
        results['Rodzaj danych:'].append(data_type)
        results['Wyznaczenie wartości:'].append(calc_type)
        
        for degree in degrees:
            if degree == 1:
                epsilon_train, epsilon_test, epsilon_train_rec, epsilon_test_rec = linear_dynamic_1_degree()
            elif degree == 2:
                epsilon_train, epsilon_test, epsilon_train_rec, epsilon_test_rec = linear_dynamic_2_degree()
            else:
                epsilon_train, epsilon_test, epsilon_train_rec, epsilon_test_rec = linear_dynamic_3_degree()
            
            if i % 2 == 0:  # Even index for "Bez rekurecji"
                if data_type == 'Dane uczące':
                    results[f'N={degree}'].append(epsilon_train)
                else:
                    results[f'N={degree}'].append(epsilon_test)
            else:  # Odd index for "Z rekurecją"
                if data_type == 'Dane uczące':
                    results[f'N={degree}'].append(epsilon_train_rec)
                else:
                    results[f'N={degree}'].append(epsilon_test_rec)
    
    df_results = pd.DataFrame(results)
    return df_results

def plot_split(y_mod_train, y_mod_test, y_mod_rec_train, y_mod_rec_test, title):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12) )
    fig.suptitle(title)
    # Plot model output vs training data
    axs[0, 0].plot(y_mod_train, label='Wyjście modelu')
    axs[0, 0].plot(y_train, label='Dane trenujące')
    axs[0, 0].set_title('Model na tle danych trenujących')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot model output vs test data
    axs[0, 1].plot(y_mod_test, label='Wyjście modelu')
    axs[0, 1].plot(y_test, label='Dane testowe')
    axs[0, 1].set_title('Model na tle danych testowych')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot recursive model output vs training data
    axs[1, 0].plot(y_mod_rec_train, label='Wyjście modelu z rekurencją')
    axs[1, 0].plot(y_train, label='Dane trenujące')
    axs[1, 0].set_title('Model rekurencyjny na tle danych trenujących')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot recursive model output vs test data
    axs[1, 1].plot(y_mod_rec_test, label='Wyjście modelu z rekurencją')
    axs[1, 1].plot(y_test, label='Dane testowe')
    axs[1, 1].set_title('Model rekurencyjny na tle danych testowych')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(f'plots/{title}.png')
    plt.show()


# def create_m_matrix(n, degree, u_data, y_data):
#     num_samples = len(u_data) - n
#     num_features = degree * n * 2
#     feature_matrix = np.zeros((num_samples, num_features))
#     for i in range(num_samples):
#         feature_row = []
#         k = i + n
#         for lag in range(1, n + 1):
#             for power in range(1, degree + 1):
#                 feature_row.append(u_data[k - lag] ** power)
#         for lag in range(1, n + 1):
#             for power in range(1, degree + 1):
#                 feature_row.append(y_data[k - lag] ** power)
#         feature_matrix[i, :] = feature_row
#     return feature_matrix

def create_columns(n_row, degree, u_train , y_train):
    columns = []
    for i in range(1, n_row + 1):
        for power in range(1, degree + 1):
            columns.append(u_train[n_row-i:-i] ** power)
            columns.append(y_train[n_row-i:-i] ** power)
    return columns


def init_model_outputs(y_train, n_row):
    y_mod_train = []
    y_mod_rec_train = []
    y_mod_test = []
    y_mod_rec_test = []

    for i in range(n_row):
        y_mod_train.append(y_train[i])
        y_mod_rec_train.append(y_train[i])
        y_mod_test.append(y_train[i])
        y_mod_rec_test.append(y_train[i])
    
    return y_mod_train, y_mod_rec_train, y_mod_test, y_mod_rec_test
def non_liner_dynamic(n_row ,degree):
    y_mod_train, y_mod_rec_train, y_mod_test, y_mod_rec_test = init_model_outputs(y_train, n_row)
    # m_matrix = create_m_matrix(n_row, degree, u_train, y_train)
    columns = create_columns(n_row, degree, u_train, y_train)
    m_matrix = np.column_stack(columns)
    w = np.linalg.lstsq(m_matrix, y_train[n_row:], rcond=None)[0]
    # y_mod_train, y_mod_rec_train, y_mod_test, y_mod_rec_test = initialize_model_outputs(y_train, y_test, n_row)
    # print(m_matrix[0])
    for k in range(len(y_train) - n_row):
        y_k = 0
        y_k_test = 0
        y_k_rek = 0
        y_k_test_rek = 0
        counter = 0
        for l in range(1, n_row + 1):
            for power in range(1, degree + 1):
                y_k += w[counter] * u_train[k+n_row-l] ** power + w[counter + 1] * y_train[k+n_row-l] ** power
                y_k_rek += w[counter] * u_train[k+n_row-l] ** power + w[counter + 1] * y_mod_rec_train[k+n_row-l] ** power
                y_k_test += w[counter] * u_test[k+n_row-l] ** power + w[counter + 1] * y_test[k+n_row-l] ** power
                y_k_test_rek += w[counter] * u_test[k+n_row-l] ** power + w[counter + 1] * y_mod_rec_test[k+n_row-l] ** power
                counter += 2
        # y_mod_train[k+n_row] = y_k
        # y_mod_rec_train[k+n_row] = y_k_rek
        # y_mod_test[k+n_row] = y_k_test
        # y_mod_rec_test[k+n_row] = y_k_test_rek
        y_mod_train.append(y_k)
        y_mod_rec_train.append(y_k_rek)
        y_mod_test.append(y_k_test)
        y_mod_rec_test.append(y_k_test_rek)

    plot_split(y_mod_train, y_mod_test, y_mod_rec_train, y_mod_rec_test, 'Model dynamiczny nieliniowy ' + str(n_row) + ' rzędu i stopnia: ' + str(degree) )
    epsilon_train = np.sum((y_mod_train - y_train) ** 2)
    epsilon_test = np.sum((y_mod_test - y_test) ** 2)
    epsilon_train_rec = np.sum((y_mod_rec_train - y_train) ** 2)
    epsilon_test_rec = np.sum((y_mod_rec_test - y_test) ** 2)

    return epsilon_train, epsilon_test, epsilon_train_rec, epsilon_test_rec 


def calculate_non_linear_errors(n_row, degree):
    results = {
        'Degree': [], 
        'Epsilon_train': [], 
        'Epsilon_test': [],
        'Epsilon_train_rec': [],
        'Epsilon_test_rec': []}
    for degree in range(1, degree + 1):
        results['Degree'].append(degree)
        epsilon_train, epsilon_test, epsilon_train_rec, epsilon_test_rec = non_liner_dynamic(n_row, degree)
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
# df_results = df_results.round(3)
# print(df_results)
# df_results.to_csv('results/dynamic_errors.csv', index=False)
print(non_liner_dynamic(3,3))
# plot_train_split()
# plot_test_split()
