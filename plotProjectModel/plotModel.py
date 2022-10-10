import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class plotModel:

    def __init__(self):
        self.dfRecTrain = None
        self.dfPrecTrain = None
        self.dfLossTrain = None
        self.dfAccuracyTrain = None
        self.dfRecTes = None
        self.dfPrecTes = None
        self.dfLossTes = None
        self.dfAccuracyTes = None
        self.model = None
        self.runTimeTrain = None
        self.runTimeTes = None

    def setRunTime(self, runTime):
        self.runTime = runTime

    def setDataModel(self, model):
        self.model = model

    def getBestAccuracy(self):
        return np.argmin([np.argmax(self.model[0]['accuracy']), np.argmax(self.model[1]['accuracy']),
                          np.argmax(self.model[2]['accuracy']), np.argmax(self.model[3]['accuracy'])])

    def getBestAccuracyTes(self):
        return np.argmax([self.model[0][1], self.model[1][1],
                          self.model[2][1], self.model[3][1]])

    def plotAccuracyTrain(self):
        # dfAccuracy = pd.DataFrame({
        #     'k1': self.modelTrain['accuracy'],
        #     'k2': self.modelTrain['loss'],
        #     'k3': self.modelTrain['precision'],
        #     'k4': self.modelTrain['recall'],
        # })
        self.dfAccuracyTrain = pd.DataFrame({
            'k1': self.model[0]['accuracy'],
            'k2': self.model[1]['accuracy'],
            'k3': self.model[2]['accuracy'],
            'k4': self.model[3]['accuracy'],
        })
        self.dfAccuracyTrain.plot(kind='line', title='Accuracy Train')

    def plotLossTrain(self):
        self.dfLossTrain = pd.DataFrame({
            'k1': self.model[0]['loss'],
            'k2': self.model[1]['loss'],
            'k3': self.model[2]['loss'],
            'k4': self.model[3]['loss'],
        })
        self.dfLossTrain.plot(kind='line', title='Loss Train')

    def plotPrecisionTrain(self):
        self.dfPrecTrain = pd.DataFrame({
            'k1': self.model[0]['precision'],
            'k2': self.model[1]['precision'],
            'k3': self.model[2]['precision'],
            'k4': self.model[3]['precision'],
        })
        self.dfPrecTrain.plot(kind='line', title='Precision Train')

    def plotRecallTrain(self):
        self.dfRecTrain = pd.DataFrame({
            'k1': self.model[0]['recall'],
            'k2': self.model[1]['recall'],
            'k3': self.model[2]['recall'],
            'k4': self.model[3]['recall'],
        })
        self.dfRecTrain.plot(kind='line', title='Recall Train')

    def plotRunTimeTrain(self, time):
        self.runTimeTrain = pd.DataFrame({
            'k1': [time[0]],
            'k2': [time[1]],
            'k3': [time[2]],
            'k4': [time[3]],
        })
        self.runTimeTrain.plot(kind='bar', title='RunTime Train')

    def plotAccuracyTes(self):
        self.dfAccuracyTes = pd.DataFrame({
            'k1': [self.model[0][1]],
            'k2': [self.model[1][1]],
            'k3': [self.model[2][1]],
            'k4': [self.model[3][1]],
        })
        self.dfAccuracyTes.plot(kind='bar', title='Accuracy Tes')

    def plotLossTes(self):
        self.dfLossTes = pd.DataFrame({
            'k1': [self.model[0][0]],
            'k2': [self.model[1][0]],
            'k3': [self.model[2][0]],
            'k4': [self.model[3][0]],
        })
        self.dfLossTes.plot(kind='bar', title='Loss Tes')

    def plotPrecisionTes(self):
        self.dfPrecTes = pd.DataFrame({
            'k1': [self.model[0][2]],
            'k2': [self.model[1][2]],
            'k3': [self.model[2][2]],
            'k4': [self.model[3][2]],
        })
        self.dfPrecTes.plot(kind='bar', title='Precision Tes')

    def plotRecallTes(self):
        self.dfRecTes = pd.DataFrame({
            'k1': [self.model[0][3]],
            'k2': [self.model[1][3]],
            'k3': [self.model[2][3]],
            'k4': [self.model[3][3]],
        })
        self.dfRecTes.plot(kind='bar', title='Recall Tes')

    def plotRunTimeTes(self, time):
        self.runTimeTes = pd.DataFrame({
            'k1': [time[0]],
            'k2': [time[1]],
            'k3': [time[2]],
            'k4': [time[3]],
        })
        self.runTimeTes.plot(kind='bar', title='RunTime Tes')

    def showPlot(self):
        plt.show()

    def writeDataAccuracyHasil(self, df, k):
        max = np.max([self.model[0][1], self.model[1][1], self.model[2][1], self.model[3][1]])
        argMax = np.argmax([self.model[0][1], self.model[1][1], self.model[2][1], self.model[3][1]])
        df['MAX'][f'Skenario {k}'] = max
        df['k'][f'Skenario {k}'] = argMax+1
        for x in range(4):
            df[f'k{x + 1}'][f'Skenario {k}'] = self.model[x][1]
        return df


