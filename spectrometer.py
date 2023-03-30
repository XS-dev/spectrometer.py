import glob

import numpy as np
import pandas as pd
import osqp
from scipy import sparse
import u_tool
import scipy as sp


class Spectrometer:

    def __init__(self, wavelength):
        self.wavelength = wavelength
        self.alm = ['Original']

    def _get_matrix_a(self, filepath,method):
        self.matrix_a = self.__get_spectral_intensity(filepath)
        if method == 1:
            matrixatemp = self.matrix_a[:, 0]
            for umtrix_a_temp in np.arange(3,121,3):
                matrixatemp = np.vstack((self.matrix_a[:, umtrix_a_temp], matrixatemp))
            # self.matrix_a = self.__absorption2transmittance(self.matrix_a)
            self.matrix_a = matrixatemp.T
        else:
            self.matrix_a = self.matrix_a[:,31:81]
        return self.matrix_a

    def _get_matrix_x(self, filepath,method,name):
        if method == 0:
            self.matrix_x = pd.DataFrame(pd.read_table(filepath, sep=',', encoding='utf-8', header=1)).values[:, 0]
            self.matrix_x = self.matrix_x[np.newaxis, :]
        else:
            self.matrix_x = np.load(name)
            self.matrix_x = self.matrix_x[np.newaxis, :]
        return self.matrix_x

    def _get_matrix_b(self,filepath,method,order,datapath):
        if method == 1:
            self.matrix_b = self.matrix_a.dot(self.matrix_x.T)*0.8
            # self.matrix_b = self.matrix_b[np.newaxis, :]
            return self.matrix_b
        elif method == 2:
            # self.matrix_b = np.load(datapath)
            mmm = pd.read_table(datapath, sep=',', encoding='utf-8', header=1)
            self.matrix_b = pd.DataFrame(mmm).values
        else:
            self.matrix_b = self.__get_spectral_intensity(filepath)
            [btempm,btempn] = self.matrix_b.shape
            btemphere = self.matrix_b[:, 0]
            for btemporderx in range(0,btempm):
                self.matrix_b[btemporderx,1:btempn+1] = self.matrix_b[btemporderx,1:btempn+1]/(btemphere[btemporderx]**2)
            self.matrix_b = self.matrix_b[:,order]*0.65
            # self.matrix_b = self.matrix_b[np.newaxis, :]
            return self.matrix_b

    def __get_spectral_intensity(self, filepath):
        filepath = glob.glob(filepath)
        f_length = len(filepath)
        dftemp = pd.DataFrame(pd.read_table(filepath[f_length - 1], sep=',', encoding='utf-8', header=1))
        df = dftemp.values[:, 0]
        for tmp in range(f_length - 2, -1, -1):
            df = np.vstack(
                (pd.DataFrame(pd.read_table(filepath[tmp], sep=',', encoding='utf-8', header=1)).values[:, 0], df))
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
        H = 2 * self.matrix_a.T.dot(self.matrix_a)
        # f=-2A'b
        f = -2. * self.matrix_a.T.dot(self.matrix_b)
        # 不等式约束条件：Ax <= b
        #  等式约束条件：A_eqx=beq


        # Define problem data
        P = sparse.csc_matrix(H)
        q = np.array(f)
        A = sparse.csc_matrix(self.matrix_a)
        l = np.array(self.matrix_b)
        u = np.array(self.matrix_b)

        # Create an OSQP object
        prob = osqp.OSQP()

        # Setup workspace and change alpha parameter
        prob.setup(P, q, A, l, u, alpha=1.4,eps_prim_inf=1e-6)

        # Solve problem
        self.xil = prob.solve().x

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
            if (j1 + j2) < epsilon:
                print("迭代次数")
                print(times)
                print("\r\n")
                self.xil = x
                # self.xil = np.linalg.pinv(self.matrix_a).dot(self.matrix_b)
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
