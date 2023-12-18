import numpy as np

def rmse(y_true, y_pred):

    soma = 0
    for i in range(0, len(y_true)):

        soma += (y_true[i] - y_pred[i])**2

    return np.sqrt(soma/len(y_true))

print(rmse(np.array([12, 35, 67]), np.array([24, 57, 37])))