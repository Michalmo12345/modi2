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


def plot_train_split():
    plt.figure(figsize=(8, 6))
    plt.scatter(u_train, y_train)
    plt.title('Train split')
    plt.xlabel('U values')
    plt.ylabel('Y values')
    plt.grid(True)
    plt.show()


def plot_test_split():
    plt.figure(figsize=(8, 6))
    plt.scatter(u_test, y_test)
    plt.title('Test split')
    plt.xlabel('U values')
    plt.ylabel('Y values')
    plt.grid(True)
    plt.show()