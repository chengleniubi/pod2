import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from my_pca import pca_analysis
from plot_gmm import plot_gmm
from utils import np2o3d, o3d2np
from visualization import display_inlier_outlier2


def pos_cor(o3d_point_cloud):
    # 姿态矫正

    # 主成分分析获得成分向量，转换成矩阵
    _, transform_para = pca_analysis(o3d_point_cloud, view=0)
    # 姿态矫正
    np_point_clouds_rotate = transform(o3d_point_cloud, transform_para, view=0)
    return np_point_clouds_rotate


def projection(np_pc_left, np_pc_right):
    # 参数投影
    np_left_2d = parametric_projection(np_pc_left, mode=0, dir_='y', view=0)
    np_right_2d = parametric_projection(np_pc_right, mode=0, dir_='y', view=0)
    return np_left_2d, np_right_2d


def index_con(list_index):
    """list_index: n*1"""
    labels = list(set(list_index))
    index = [np.where(list_index == labels[i]) for i in range(len(labels))]
    return index


def separate(np_point_clouds, view=0):
    # 分离
    # TODO: 聚类方法总结
    np_points_2d = parametric_projection(np_point_clouds, mode=0, dir_='x', view=0)

    # 高斯方法  会产生类别随机性
    list_ = plot_gmm(np_points_2d, n_com=2, mode=0, view=0)

    index = index_con(list_)
    point_cloud1 = np_point_clouds[index[0]]
    point_cloud2 = np_point_clouds[index[1]]

    # 消除类别随机性
    a = np.mean(point_cloud1, axis=0)
    b = np.mean(point_cloud2, axis=0)

    point_left = point_cloud2 if a[1] > b[1] else point_cloud1
    point_right = point_cloud1 if a[1] > b[1] else point_cloud2

    if view:
        display_inlier_outlier2(point_cloud1, point_cloud2)

    return point_left, point_right


def parametric_projection(np_point_clouds, mode=0, dir_='x', view=0):
    # 参数投影
    def projection_t(np_point_clouds_t, dir_t='x'):
        np_points_2d_t = np.zeros((np_point_clouds_t.shape[0], 2))
        if dir_t is 'x':
            np_points_2d_t[:, 0] = np_point_clouds_t[:, 1]
            np_points_2d_t[:, 1] = np_point_clouds_t[:, 2]
        elif dir_t is 'y':
            np_points_2d_t[:, 0] = np_point_clouds_t[:, 0]
            np_points_2d_t[:, 1] = np_point_clouds_t[:, 2]
        else:
            np_points_2d_t[:, 0] = np_point_clouds_t[:, 0]
            np_points_2d_t[:, 1] = np_point_clouds_t[:, 1]
        return np_points_2d_t

    def show_t(np_points_2d_t):
        temp_points_2d = np.zeros((np_point_clouds.shape[0], 3))
        temp_points_2d[:, :2] = np_points_2d_t[:, :]
        o3d_temp_points = np2o3d(temp_points_2d)
        o3d.visualization.draw_geometries([o3d_temp_points])
        plt.scatter(np_points_2d_t[:, 0], np_points_2d_t[:, 1])
        plt.show()

    if mode is 0:
        if dir_ is 'x':
            np_points_2d = projection_t(np_point_clouds, dir_t='x')
            if view:
                show_t(np_points_2d)
        elif dir_ is 'y':
            np_points_2d = projection_t(np_point_clouds, dir_t='y')
            if view:
                show_t(np_points_2d)
        else:
            np_points_2d = projection_t(np_point_clouds, dir_t='z')
            if view:
                show_t(np_points_2d)
        return np_points_2d


def transform(o3d_point_clouds, transform_para, view=0):
    np_point_clouds = o3d2np(o3d_point_clouds)
    np_point_rotate = np_point_clouds.dot(transform_para)
    if view:
        o3d_point_rotate = np2o3d(np_point_rotate)
        pca_analysis(o3d_point_rotate, 1)
    return np_point_rotate
