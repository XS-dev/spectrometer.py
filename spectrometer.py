import glob

import numpy as np
import pandas as pd

import u_tool


class Spectrometer:

    def __init__(self, wavelength):
        self.wavelength = wavelength
        self.alm = ['Original']

    def _get_matrix_a(self, filepath):
        self.matrix_a = self.__get_spectral_intensity(filepath)
        self.matrix_a = self.__absorption2transmittance(self.matrix_a)
        return self.matrix_a

    def _get_matrix_x(self, filepath):
        self.matrix_x = pd.DataFrame(pd.read_table(filepath, sep=',', encoding='gbk', header=1)).values[:, 1]
        self.matrix_x = self.matrix_x[np.newaxis, :]
        return self.matrix_x

    def _get_matrix_b(self):
        self.matrix_b = self.matrix_a.dot(self.matrix_x.T)
        return self.matrix_b

    def __get_spectral_intensity(self, filepath):
        filepath = glob.glob(filepath)
        f_length = len(filepath)
        df = pd.DataFrame(pd.read_table(filepath[f_length - 1], sep=',', encoding='gbk', header=1)).values[:, 1]
        for tmp in range(f_length - 2, -1, -1):
            df = np.vstack(
                (pd.DataFrame(pd.read_table(filepath[tmp], sep=',', encoding='gbk', header=1)).values[:, 1], df))
        self.matrix = df
        return df

    def __absorption2transmittance(self, matrix):
        self.matrix = 1 / np.power(10, matrix)
        return self.matrix

    def _restoration(self, algorithm):
        match algorithm:
            case "ALM":
                self.alm.append('ALM')
                self.__simple_ALM()
            case "LS":
                self.__simple_LS()

    def __simple_LS(self):
        # 定义二次规划函数所需参数
        # H = 2A'A
        # H = 2 * self.matrix_a.T.dot(self.matrix_a)
        # f=-2A'b
        # f = -2. * self.matrix_a.T.dot(self.matrix_b)
        # 不等式约束条件：Ax <= b
        #  等式约束条件：A_eqx=beq
        # 二次规划函数

        print('LS')

    def __simple_ALM(self):
        # 初始值准备
        A = self.matrix_a
        b = self.matrix_b
        G = u_tool.create_1d_gradient(len(self.wavelength))

        # 罚因子
        mu_1 = 10
        mu_2 = 10

        rho = 1.01
        # mu_1max = 100000
        # mu_2max = 100000

        y_1 = 0
        y_2 = 0

        epsilon = 1e-5

        x = np.abs(np.sin(self.wavelength / 30))
        x = x[np.newaxis, :].T

        times = 0
        while 1:
            times = times + 1
            g = np.int_((G.dot(x) + y_1 / mu_1) > (1 / mu_1)) * ((G.dot(x) + y_1 / mu_1) - 1 / mu_1) + (
                    (G.dot(x) + y_1 / mu_1) + 1 / mu_1) * np.int_(((G.dot(x) + y_1 / mu_1) < (1 / -mu_1)))
            x1 = mu_1 * (G.T.dot(G)) + mu_2 * ((A.T).dot(A))
            x_tmp = G.T.dot(G)
            x2 = mu_1 * (G.T.dot((g - y_1 / mu_1))) + mu_2 * ((A.T).dot(b - y_2 / mu_2))
            x = np.linalg.inv(x1).dot(x2)
            j1 = np.linalg.norm(G.dot(x) - g)
            j2 = np.linalg.norm(A.dot(x)-b)
            if ( j1 + j2) < epsilon:
                print("迭代次数")
                print(times)
                print("\r\n")
                self.xil = x
                self.matrix_x = np.vstack((self.matrix_x,x.T))
                return x
            else:
                # 迭代y1
                y_1 = mu_1 * (G.dot(x) - g) + y_1
                # 迭代y2
                y_2 = mu_2 * (A.dot(x) - b) + y_2
                # 迭代μ1
                mu_1 = rho * mu_1
                # mu_1 = min(rho * mu_1, mu_1max);
                # 迭代μ2
                mu_2 = rho * mu_2
                # mu_2 = min(rho * mu_2, mu_2max);
