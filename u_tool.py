import numpy as np


def create_1d_gradient(row):
    if row <= 1:
        return 0
    else:
        gradient = np.zeros((row, row))
        for index in range(0, row):
            if index == 0:
                gradient[index, index] = -1
                gradient[index, index + 1] = 1
            elif index == row-1:
                gradient[index, index - 1] = -1
                gradient[index, index] = 1
            else:
                gradient[index , index - 1] = -0.5
                gradient[index , index+1] = 0.5
    return gradient

def create_1d_differential(row):
    if row <= 1:
        return 0
    else:
        gradient = np.zeros((row, row))
        for index in range(0, row):
            if index == row-1:
                gradient[index, index] = -1
            else:
                gradient[index , index] = -1
                gradient[index , index+1] = 1

                # gradient[index - 1, index - 1] = -1
                # gradient[index - 1, index] = 1
    return gradient

#[mae, mape, mse, rmse]
def deviation_calculation(reference,observation):
    # 计算平均绝对误差
    mae = np.mean(np.abs(observation-reference))
    # 计算平均绝对百分比误差（真实值不能有0）
    mape = np.mean(np.abs(observation-reference) / reference)
    # 计算均方误差
    mse = np.mean((observation-reference)**2)
    # 计算均方根误差
    rmse = np.sqrt(mse)
    return [mae,mape,mse,rmse]