'''
Project: Perceptron Simples
Created: 15/04/2019
@author: João Victor
'''

from pandas.core.frame import DataFrame,Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Constant of expression w_new = w_old + e*N*x
N = 0.1

def normalize(data):
    for col in data.columns:
        min = np.min(data[col])
        max = np.max(data[col])
        data[col] = [(data.at[i,col] - min)/(max-min)
                        for i in range(len(data))]
    return data

def trainTest(input_vectors = None, class_d = None):
        train_x,test_x = _splitData_(input_vectors)
        train_d = class_d.iloc[train_x.index]
        test_d = class_d.drop(train_d.index)

        return train_x,test_x,train_d,test_d

def _splitData_(data):
    index_random = _randomIndex_(data,0.8)
    return data.loc[index_random],data.drop(index_random)

def _randomIndex_(data,count):
    random_count = int(len(data) * count)
    return np.random.choice(data.index,random_count,replace = False)

#Simple Perceptron
class Perceptron(object):
    __slots__ = ['vector_peso']

    def __init__(self):
        self.vector_peso = None

    def trainingModel(self,data,class_d):
        self.vector_peso = np.random.random(1 + data.shape[1])

        for count in range(100):
            index_random = _randomIndex_(data,1)
            data = data.loc[index_random]
            class_d = class_d.loc[index_random]

            for index,row in data.iterrows():
                train_x = np.array([-1] + row.tolist())
                func_u = np.inner(train_x,self.vector_peso)
                class_y = 1 if func_u >= 0 else 0
                error = class_d[index] - class_y
                self.vector_peso = self.vector_peso + N * error * train_x

    def predict(self,vector_input):
        predict_list = []
        if type(vector_input) == list:
            vector = np.array([-1] + vector_input)
            func_u = np.inner(vector,self.vector_peso)
            return True if func_u >= 0 else False

        for index,row in vector_input.iterrows():
            vector = np.array([-1] + row.tolist())
            func_u = np.inner(vector,self.vector_peso)
            predict_list += [True if func_u >= 0 else False]

        return predict_list

    def getPesos(self):
        return self.vector_peso

class Statistic(object):
    @staticmethod
    def holdOut(model,input_vectors,class_d,realization_count = 20):
        realizations_list = []
        for realization in range(realization_count):
            train_x,test_x, train_d,test_d = trainTest(input_vectors,class_d)
            model.trainingModel(train_x,train_d)
            realizations_list += [Statistic.hitRate(model,test_x,test_d)]

        return realizations_list

    @staticmethod
    def hitRate(model,test_data,test_class_d):
        predict_y = model.predict(test_data)
        return (predict_y == test_class_d).mean()

    @staticmethod
    def accuracy(model,input_vectors,class_d,realization_count = 20):
        realizations = Statistic.holdOut(model,
                                        input_vectors,
                                        class_d,
                                        realization_count)
        return np.mean(realizations)

    @staticmethod
    def std(model,input_vectors,class_d,realization_count = 20):
        realizations = Statistic.holdOut(model,
                                        input_vectors,
                                        class_d,
                                        realization_count)
        return np.std(realizations)

    @staticmethod
    def matrix_of_confusion(model,input_vectors,class_d):
        matriz_confusion = np.array( [[0,0],[0,0]])
        class_y = model.predict(input_vectors)
        class_d.index = range(len(class_y)) #reset index of class d

        for index in range(len(class_y)):
            y = int(class_y[index])
            d = int(class_d[index])
            matriz_confusion[d][y] += 1

        return matriz_confusion

    @staticmethod
    def decision_graph(model):
        class_true=[]
        class_false=[]
        for x1 in np.arange(0,1.0,0.01):
            for x2 in np.arange(0,1.0,0.01):
                vector = [x1,x2]
                y = model.predict(vector)
                if y:
                    plt.scatter(x1,x2,c = "#b2f441")
                else:
                    plt.scatter(x1,x2,c = "#f4d341")
        plt.title("Gráfico de Decisão")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()
