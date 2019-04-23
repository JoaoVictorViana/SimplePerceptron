'''
Project: Perceptron Simples
Created: 15/04/2019
@author: João Victor
'''

import pandas as pd
from Perceptron import *

def Main():
    #Load Data
    iris_df = pd.read_csv("Data/Iris_Data.csv")

    #Clear collumn 'species'
    iris_df['species'] = iris_df['species'].apply(
        lambda x: 1 if 'setosa' in x else 0
    )

    #Separate the class 'd' from the input vectors
    class_d = iris_df.species
    iris_df.drop('species', axis = 1,inplace = True)

    #Split train from the test
    train_x,test_x, train_d,test_d = trainTest(iris_df,class_d)

    #Create the Perceptron
    perceptron = Perceptron()
    perceptron.trainingModel(train_x,train_d)

    print(perceptron.predict([6.5,3.0,5.0,1.2])) #--> Return False
    print(perceptron.getPesos())
    print(Statistic.accuracy(perceptron,iris_df,class_d))

if __name__ == '__main__':
    Main()
