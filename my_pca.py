import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from draw_geometry import draw_axis
from utils import o3d2np


def normalization(np_points):
    mo = np.mean(np_points, axis=0)
    return np_points - mo, mo


def my_PCA(data, sort=True):
    """ Calculate PCA for given point cloud
    功能：计算PCA的函数
    输入：
    data：点云，NX3的矩阵
    sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
    输出：
    eigenvalues：特征值
    eigenvectors：特征向量
    """
    # 归一化
    X_normalized, _ = normalization(data)
    # 构造协方差矩阵
    H = np.dot(X_normalized.T, X_normalized)
    # SVD分解,计算方形矩阵特征值和特征向量
    eigenvectors, eigenvalues, eigenvectors_t = np.linalg.svd(H)
    for i in range(np.shape(eigenvectors)[0]):
        if eigenvectors[i, i] < 0:
            eigenvectors[:, i] = -eigenvectors[:, i]

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors


def sklearn_pca(o3d_point_clouds, n_components=3, view=0):
    # sklearn中的PCA
    pca = PCA(n_components=n_components)
    # newX = pca.fit_transform(X)
    pca.fit(o3d_point_clouds)

    if view:
        print('pca.explained_variance_ratio_: ', pca.explained_variance_ratio_)
        print('pca.explained_variance_:', pca.explained_variance_)

        print('pca.n_components_: ', pca.n_components_)
        print('pca.components_: ', pca.components_)
        np_cut_points_new = pca.transform(o3d_point_clouds)

        plt.scatter(np_cut_points_new[:, 0], np_cut_points_new[:, 1], marker='o')
        plt.show()
    return pca.explained_variance_, pca.components_.T


def pca_analysis(o3d_point_clouds, view=0):
    # PCA主成分分析
    np_point_clouds = o3d2np(o3d_point_clouds)
    v, u = my_PCA(np_point_clouds)  # 代码运行时长：0.001s
    # v, u = sklearn_pca(np_point_clouds)  # 代码运行时长：0.013s

    # 罗德里格旋转公式 # 获得旋转矩阵
    # transform = np.vstack(u[:, 0], u[:, 1], u[:, 2]).T
    transform = np.array((u[:, :3]))  # 0.000003

    if view:
        print('v:', v)
        print('the main orientation of this point_cloud is:\n ', u[:, 0], u[:, 1], u[:, 2])
        print('transform:\n', transform)
        # 画出PCA主方向
        line_set = draw_axis(u, long=20)
        # 画坐标轴
        line_set1_wcs = draw_axis(u=None, long=10)
        # 可视化
        o3d.visualization.draw_geometries([o3d_point_clouds, line_set, line_set1_wcs])
    return u, transform


if __name__ == '__main__':
    from filter import Pretreatment

    filename_ = '1point1010.ply'
    o3d_point_cloud = Pretreatment(filename_)
    pca_analysis(o3d_point_cloud, view=0)
