'''
Project: Perceptron Simples
Created: 15/04/2019
@author: João Victor
'''

from pandas.core.frame import DataFrame,Series
import pandas as pd
import numpy as np

#Constant of expression w_new = w_old + e*N*x
N = 0.1

def trainTest(input_vectors = None, class_d = None):
    if type(input_vectors) != DataFrame:
        raise Exception("Tipo de dado inválido, só é permitido DataFrame.")

    if type(class_d) != Series:
        raise Exception("Tipo de dado inválido, só é permitido Series.")

    if len(input_vectors) != len(class_d):
        raise Exception("O tamanho dos dois dados precisam ser iguais.")

    if input_vectors.empty == False and class_d.empty == False:
        train_x,test_x = _splitData_(input_vectors)
        train_d = class_d.iloc[train_x.index]
        test_d = class_d.drop(train_d.index)

        return train_x,test_x,train_d,test_d
    else:
        raise Exception("Não é possível passar NoneType como parâmetro.")

def _splitData_(data):
    index_random = _randomSample_(data,0.8)
    return data.loc[index_random],data.drop(index_random)

def _randomSample_(data,count):
    random_count = int(len(data) * count)
    return np.random.choice(data.index,random_count,replace = False);

#Simple Perceptron
class Perceptron(object):
    __slots__ = ['vector_peso']

    def __init__(self):
        self.vector_peso = None

    def trainingModel(self,data,class_d):
        self.vector_peso = np.random.random(1 + data.shape[1])

        for count in range(100):
            index_random = _randomSample_(data,1)
            data = data.loc[index_random]
            class_d = class_d.loc[index_random]

            index_count = 0
            for index,row in data.iterrows():
                train_input_vector = np.array([-1] + row.tolist())
                func_u = np.inner(train_input_vector,self.vector_peso)
                class_y = 1 if func_u >= 0 else 0
                error = class_d.iat[index_count] - class_y
                index_count += 1
                self.vector_peso = (self.vector_peso +
                                        N * error *
                                             train_input_vector)


    def predict(self,vector_input):
        predict_list = []
        if type(vector_input) == list:
            vector = np.array([-1] + vector_input)
            func_u = np.inner(vector,self.vector_peso)
            return True if func_u >= 0 else False

        for number in range(len(vector_input)):
            vector = np.array([-1] + vector_input.iloc[number].tolist())
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
        realizations = Statistic.holdOut(model,input_vectors,class_d,realization_count)
        return np.mean(realizations)
