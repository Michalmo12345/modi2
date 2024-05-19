import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data_train = pd.read_csv('data/danedynucz18.txt', delimiter=r"\s+", header=None, names=['U', 'Y'])
data_test = pd.read_csv('data/danedynwer18.txt', delimiter=r"\s+", header=None, names=['U', 'Y'])


u_data_train = data_train['U']
y_data_train = data_train['Y']
u_data_test = data_test['U']
y_data_test = data_test['Y']
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

def create_columns(n_row, degree, u_train , y_train):
    columns = []
    for i in range(1, n_row + 1):
        for power in range(1, degree + 1):
            columns.append(u_train[n_row-i:-i] ** power)
            columns.append(y_train[n_row-i:-i] ** power)
    return columns

def non_liner_dynamic(n_row, degree):
    y_mod_train, y_mod_rec_train, y_mod_test, y_mod_rec_test = initialize_model_outputs(y_train, y_test, n_row)
    columns = create_columns(n_row, degree, u_train, y_train)
    m_matrix = np.column_stack(columns)
    w = np.linalg.lstsq(m_matrix, y_train[n_row:], rcond=None)[0]

    for k in range(len(y_train) - n_row):
        y_k = 0
        y_k_test = 0
        y_k_rek = 0
        y_k_test_rek = 0
        counter = 0
        for l in range(1, n_row + 1):
            for power in range(1, degree + 1):
                u_train_val = u_train[k + n_row - l]
                y_train_val = y_train[k + n_row - l]
                u_test_val = u_test[k + n_row - l]
                y_test_val = y_test[k + n_row - l]
                y_mod_rec_train_val = y_mod_rec_train[k + n_row - l]
                y_mod_rec_test_val = y_mod_rec_test[k + n_row - l]

                y_k += w[counter] * u_train_val ** power + w[counter + 1] * y_train_val ** power
                y_k_rek += w[counter] * u_train_val ** power + w[counter + 1] * y_mod_rec_train_val ** power
                y_k_test += w[counter] * u_test_val ** power + w[counter + 1] * y_test_val ** power
                y_k_test_rek += w[counter] * u_test_val ** power + w[counter + 1] * y_mod_rec_test_val ** power
                counter += 2

        y_mod_train[k + n_row] = y_k
        y_mod_rec_train[k + n_row] = y_k_rek
        y_mod_test[k + n_row] = y_k_test
        y_mod_rec_test[k + n_row] = y_k_test_rek

    title = f'Model_dynamiczny_nieliniowy_{n_row}_rzędu_i_stopnia_{degree}'
    plot_split(y_mod_train, y_mod_test, y_mod_rec_train, y_mod_rec_test, title)
    epsilon_train = np.sum((y_mod_train - y_train) ** 2)
    epsilon_test = np.sum((y_mod_test - y_test) ** 2)
    epsilon_train_rec = np.sum((y_mod_rec_train - y_train) ** 2)
    epsilon_test_rec = np.sum((y_mod_rec_test - y_test) ** 2)

    return epsilon_train, epsilon_test, epsilon_train_rec, epsilon_test_rec 

def calculate_non_linear_errors():
    # degrees = ['drugi', 'trzeci', 'czwarty', 'piąty', 'szósty']
    degrees  = list(range(9, 10))
    dynamic_orders = list(range(13, 30))

    columns = pd.MultiIndex.from_product([degrees, dynamic_orders], names=['Stopień:', 'Rząd dyn.'])

    index = pd.MultiIndex.from_product([['Ucz', 'Wer'], ['Bez rek.', 'Z rek.']], names=['Dane:', 'Typ:'])


    df = pd.DataFrame(np.nan, index=index, columns=columns)
    for stopien, rzad in df.columns:
        n_row = rzad
        # degree_mapping = {'drugi': 2, 'trzeci': 3, 'czwarty': 4, 'piąty': 5, 'szósty': 6}
        # degree = degree_mapping[stopien]
        degree = stopien
        epsilon_train, epsilon_test, epsilon_train_rec, epsilon_test_rec = non_liner_dynamic(n_row, degree)
        
        df.loc[('Ucz', 'Bez rek.'), (stopien, rzad)] = epsilon_train
        df.loc[('Ucz', 'Z rek.'), (stopien, rzad)] = epsilon_train_rec
        df.loc[('Wer', 'Bez rek.'), (stopien, rzad)] = epsilon_test
        df.loc[('Wer', 'Z rek.'), (stopien, rzad)] = epsilon_test_rec

    return df.round(2)

def highlight_df(df):
    return df.style.apply(lambda x: ['background: lightblue' if x.name[0] == 'Ucz' else '' for i in x], axis=1)
# plot_train_split()
# plot_test_split()
# linear_dynamic_1_degree()
# linear_dynamic_2_degree()
# linear_dynamic_3_degree()
df_results = calculate_dynamic_errors()
df_results = df_results.round(3)
print(df_results)
df_results.to_csv('results/dynamic_errors.csv', index=False)
# print(non_liner_dynamic(3,3))
# print(non_liner_dynamic(10,10))
# plot_train_split()
# plot_test_split()
# print(calculate_non_linear_errors())
print(non_liner_dynamic(15, 9))
# df = calculate_non_linear_errors()
# print(df)
# df.to_csv('results/non_linear_dynamic_errors_9.csv')