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
        train_input_vector,test_input_vector = _splitData_(input_vectors)
        train_class_d,test_class_d = _splitData_(class_d)

        return train_input_vector,test_input_vector,train_class_d,test_class_d
    else:
        raise Exception("Não é possível passar NoneType como parâmetro.")

def _splitData_(data):
    split_count = int(len(data) * 0.8)
    index_random = np.random.choice(len(data),
                                    split_count,
                                    replace = False);

    return data.loc[index_random],data.drop(index_random)

#Simple Perceptron
class Perceptron(object):
    __slots__ = ['vector_peso']

    #def trainModel():
