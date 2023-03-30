# coding = utf-8
if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt

    import spectrometer as spec

    # 定义测量数据目录路径
    # dirPath = './raw_data/absorption/CDS/*.txt'
    Apath = './Adata/*.txt'
    # 定义原始数据文件路径
    Xpath = './Adata/533_2349.txt'


    bPath = './Bdata/*.txt'
    # 定义波长范围
    # wavelength = np.arange(400, 521, 3)
    wavelength = np.arange(450, 500, 1)

    # 实例化类
    u_matrix = spec.Spectrometer(wavelength)
    #量子点数量
    dotnums = 16
    # 获得Ax=b中的各个矩阵
    u_matrix._get_matrix_a(Apath,0)
    u_matrix._get_matrix_x(Xpath,1,'test1.npy')
    u_matrix._get_matrix_b(bPath,1,75)

    u_matrix._restoration("LS")

    # 绘制图像
    fig = plt.figure(figsize=(12, 6), dpi=200)

    ax = fig.add_subplot(121)
    plt_type = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    for plt_tmp in range(0, dotnums):
        plt.plot(wavelength, u_matrix.matrix_a[plt_tmp, :])

    ax = fig.add_subplot(122)
    # for plt_tmp in range(0, 1):
    #     plt.scatter(wavelength, u_matrix.matrix_x[plt_tmp, :], marker=plt_type[plt_tmp], label=u_matrix.alm[plt_tmp], s=4)
    plt.plot(wavelength,u_matrix.xil,label = "qsop",c = 'red')
    plt.xlabel("wavelength/nm", fontdict={'size': 12})
    plt.ylabel("Transmittance", fontdict={'size': 12})
    plt.title("Comparison", fontdict={'size': 15})

    plt.legend(loc='best')
    plt.savefig("check_1.svg")
    plt.show()
