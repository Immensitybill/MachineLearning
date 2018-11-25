
import pandas as pd

import matplotlib.pyplot as plt


class Scatter:

    @staticmethod
    def run():
        Scatter.draw(Scatter.readData())

    @staticmethod
    def readData():
        dataPath = "..\logisticRegression\ex2data2.txt"
        data = pd.read_csv(dataPath, header=None, names=['test 1', 'test 2', 'Admitted'])
        return data

    @staticmethod
    def draw(data):
        positive = data[data['Admitted'].isin([1])]
        negative = data[data['Admitted'].isin([0])]

        fig,ax =plt.subplots(figsize=(12,8))

        ax.scatter(positive['test 1'], positive['test 2'], s=50, c='b', marker='o', label='Admitted')
        ax.scatter(negative['test 1'], negative['test 2'], s=50, c='r', marker='x', label='Not Admitted')
        plt.grid(True)
        plt.show()

Scatter.run()




