import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('data/danestat18.txt', delimiter=r"\s+", header=None, names=['U', 'Y'])

u_data = data['U']
y_data = data['Y']
u_train, u_test, y_train, y_test = train_test_split(u_data, y_data, test_size=0.5, random_state=42)
u_train = np.array(u_train)
u_test = np.array(u_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


def plot_train_and_test_split(u_train, y_train, u_test, y_test):

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  

    axs[0].scatter(u_train, y_train)
    axs[0].set_title('Dane uczące')
    axs[0].set_xlabel('u')
    axs[0].set_ylabel('y')
    axs[0].grid(True)


    axs[1].scatter(u_test, y_test, color='orange')
    axs[1].set_title('Dane weryfikujące')
    axs[1].set_xlabel('u')
    axs[1].set_ylabel('y')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig('plots/data_visualization__static.png')
    plt.show()


def plot_static_data(u_train, y_train, u_test, y_test):
    plt.scatter(u_data, y_data,s=10)
    plt.title('Dane statyczne')
    plt.xlabel('u')
    plt.ylabel('y')
    plt.grid(True)
    plt.savefig('plots/all_data_visualization_static.png')
    plt.show()

# plot_train_and_test_split(u_train, y_train, u_test, y_test)
# plot_static_data(u_train, y_train, u_test, y_test)
