
import numpy as np
import matplotlib.pyplot as plt

class AllPlot():
    def __init__(self, figsize):
        self.figsize = figsize

    def timeplot(self, pred, fact):
        assert len(pred) == len(fact), "lenght not match!"
        F = len(pred)
        xticks = np.arange(F)
        plt.figure(figsize=self.figsize)
        plt.plot(xticks, pred, 'x', label='predict')
        plt.plot(xticks, fact, 's', label='observed')
        plt.legend()
        plt.title('Forecasts for counterfactual')
        plt.xlabel('Time step')
        plt.show()
