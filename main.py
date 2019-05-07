'''
Project: Perceptron Simples
Created: 15/04/2019
@author: João Victor
'''
import pandas as pd
from Perceptron import *
import seaborn as sbn
import matplotlib.pyplot as plt

def Main():
    #Load Data Íris
    iris_df = pd.read_csv("Data/Iris_Data.csv")

    #Clear collumn 'species'
    iris_df['species'] = iris_df['species'].apply(
        lambda x: 1 if 'virginica' in x else 0
    )

    #Separate the class 'd' from the input vectors
    class_iris = iris_df.species
    iris_df.drop(['species'],axis = 1,inplace = True)

    #Artificial
    part_1 = np.random.randint(-10,10,size=20) * 0.01
    part_2a = np.array([np.random.randint(-10,10,size=10) * 0.01])
    part_2b = np.array([np.random.randint(90,99,size=10) * 0.01])
    part_3Sa = np.array([np.random.randint(90,99,size=10) * 0.01])
    part_3b = np.array([np.random.randint(-10,10,size=10) * 0.01])
    part_4 = np.random.randint(90,99,size=20) * 0.01

    part_1.shape = (10,2)
    part_2a.shape = (10,1)
    part_2b.shape = (10,1)
    part_3a.shape = (10,1)
    part_3b.shape = (10,1)
    part_4.shape = (10,2)

    class_0 = np.zeros(30)
    class_1 = np.ones(10)

    part_2 = np.concatenate((part_2a,part_2b),axis = 1)
    part_3 = np.concatenate((part_3a,part_3b),axis = 1)

    artificial = np.concatenate((part_1,part_2,part_3,part_4))
    artificial = artificial.reshape(40,2)

    class_artificial = pd.Series(np.concatenate((class_0,class_1)))
    artificial_df = pd.DataFrame(artificial, columns = ['x1','x2'])

    #Split train from the test
    train_x,test_x, train_d,test_d = trainTest(iris_df,class_iris)

    #Create the Perceptron
    perceptron = Perceptron()
    perceptron.trainingModel(train_x,train_d)
    #print(perceptron.predict([x1,x2]))
    #print(Statistic.accuracy(perceptron,iris_df,class_iris))
    #print(Statistic.std(perceptron,iris_df,class_iris))
    #print(Statistic.matrix_of_decision(perceptron,test_x,test_d))
    #Statistic.decision_graph(perceptron)

if __name__ == '__main__':
    Main()
