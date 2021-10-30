import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sympy import symbols, diff
from Feature_processing import index_con
from my_curve_fit import fitting
# from utils import visualization_point_cloud, bubbleSort_y_3
from visualization import display_inlier_outlier2


def batch_simple_x(pointcloud, section=1.0):
    x_min = math.floor(min(pointcloud[:, 0]))
    num_section = [(pointcloud[i, 0] - x_min) // section for i in range(pointcloud.shape[0])]
    index_simple = index_con(num_section)
    return index_simple


def batch_simple_y(pointcloud, section=1.0):
    x_max = math.ceil(max(pointcloud[:, 1]))
    index_simple = []
    for i in range(pointcloud.shape[0]):
        if (x_max - pointcloud[i, 1]) < section:
            index_simple.append(pointcloud[i, 2])
    return index_simple


def batch_simple(points_2d, section_x=5.0, section_y=5.0, view=0):
    """output:simple_point[x,y,index]"""
    points_xyindex = np.array([[points_2d[i, 0], points_2d[i, 1], i] for i in range(points_2d.shape[0])])
    index_simple_x = batch_simple_x(points_xyindex, section=section_x)

    index_simple = []
    for i in range(len(index_simple_x)):
        a = points_xyindex[index_simple_x[i]]
        index_simple_y = batch_simple_y(a, section=section_y)
        b = np.array(index_simple_y).astype(int)
        index_simple.append(b)

    index_simple = np.concatenate(index_simple, axis=0)
    view_point = points_2d[index_simple]
    # simple_points = []  # 设定采集点储存点
    #
    # for i in range(len(index_simple)):
    #     # 冒泡排序
    #     ggg = bubbleSort_y_3(points_2d[index_simple[i]])
    #     for j in range(ggg.shape[0] if sim_num > ggg.shape[0] else sim_num):
    #         simple_points.append(ggg[j])
    # simple_points = np.array(simple_points)
    if view:
        plt.scatter(points_2d[:, 0], points_2d[:, 1])
        plt.scatter(view_point[:, 0], view_point[:, 1])
        plt.show()
        # visualization_point_cloud(simple_points, '2d')
    return view_point, index_simple


def gudi(simple_points, ff):
    # 谷底算法
    ppoint = []

    for i in np.linspace(min(simple_points[:, 0]), max(simple_points[:, 0]), 100, endpoint=True):
        if np.logical_and(np.logical_and(ff(i) < ff(i - 2), ff(i) < ff(i - 4)),
                          np.logical_and(ff(i) < ff(i + 2), ff(i) < ff(i + 4))):
            if not ppoint:
                ppoint.append(i)
            else:
                if abs(i - ppoint[-1]) > 5:
                    ppoint.append(i)
                else:
                    ppoint[-1] = (i + ppoint[-1]) / 2
        else:
            pass
    print(ppoint)

    plt.plot(ppoint[:], ff(ppoint[:]), 'o', markersize=20)
    plt.show()


def gudi2(simple_points, ff, step=5, accuracy_step=0.5, view=0):
    # 谷底算法
    x = symbols('x')
    ff1_fiff = diff(ff(x), x)

    all_min_point_x = []
    for i in np.arange(np.around(min(simple_points[:, 0])), np.around(max(simple_points[:, 0])), step):
        if np.logical_and(ff1_fiff.subs('x', i) < 0, ff1_fiff.subs('x', i + step) > 0):
            for j in np.arange(i, i + step, accuracy_step):
                if np.logical_and(ff1_fiff.subs('x', j) < 0, ff1_fiff.subs('x', j + accuracy_step) > 0):
                    all_min_point_x.append(j + accuracy_step / 2)
                elif ff1_fiff.subs('x', j + accuracy_step) == 0:
                    all_min_point_x.append(j + accuracy_step)
        elif ff1_fiff.subs('x', i + step) == 0:
            all_min_point_x.append(i + step)

    # 可视化
    if view:
        plt.plot(all_min_point_x[:], ff(all_min_point_x[:]), 'o', markersize=10)
        plt.show()

    return np.array(all_min_point_x), ff(all_min_point_x)


def find_nbrs_2d(points_xy, all_min_point_x, all_min_point_z, num=20, view=0):
    # 插值求映射
    # 最近邻
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(points_xy)
    p_index = [nbrs.kneighbors([[all_min_point_x[i], all_min_point_z[i]]], n_neighbors=num, return_distance=False)
               for i in range(all_min_point_x.shape[0])]
    points_nbrs = [[points_xy[np.squeeze(np.array(p_index[i]))]] for i in range(len(p_index))]

    """# 画圆
    neigh = NearestNeighbors(radius=8, algorithm='ball_tree').fit(simple_points_xy)
    # rng = neigh.radius_neighbors([[all_min_point_x[2], all_min_point_z[2]]])
    # print(rng)
    pindex1 = []
    for i in range(all_min_point_x.shape[0]):
        index = neigh.radius_neighbors([[all_min_point_x[i], all_min_point_z[i]]], return_distance=False)
        pindex1.append(index[0])
    print(pindex1)
    print(pindex1[0])
    pindex1 = np.concatenate(pindex1, axis=0)
    print(np.array(pindex1))
    # nbrs = neigh.radius_neighbors([[0, 0, 1.3]], 0.4, return_distance=False)"""
    if view:
        plt.scatter(points_xy[:, 0], points_xy[:, 1])
        for i in range(len(points_nbrs)):
            point_view = np.squeeze(np.array(points_nbrs[i]))
            plt.plot(point_view[:, 0], point_view[:, 1], 'o', markersize=10)
            pass
        plt.show()
    return p_index, points_nbrs


def find_nbrs_3d(points_xyz, mid_point, radius=1, view=0):
    # 球最近邻
    nbrs = NearestNeighbors().fit(points_xyz)
    p_index = [nbrs.radius_neighbors([mid_point[i]], radius=radius, return_distance=False)
               for i in range(mid_point.shape[0])]
    p_index_tmp = np.concatenate(np.squeeze(p_index))
    in_index = []
    for i in range(points_xyz.shape[0]):
        if i not in p_index_tmp:
            in_index.append(i)
    points_nbrs = points_xyz[p_index_tmp]
    points_backend = points_xyz[in_index]
    if view:
        display_inlier_outlier2(points_backend, points_nbrs)
    return p_index, points_nbrs


def interpolation_mapping(index_left, np_pc_left, view=0):
    points_nbrs = [[np_pc_left[np.squeeze(np.array(index_left[i]))]] for i in range(len(index_left))]

    if view:
        point = np.squeeze(np.concatenate(points_nbrs, 1))
        display_inlier_outlier2(np_pc_left, point)
    return points_nbrs


def simple_deal(np_left_2d):
    # batch采样
    simple_points_xyl, _ = batch_simple(np_left_2d, section_x=1, section_y=4, view=0)
    # np.savetxt('simple_points2.txt', simple_points_xyL)

    # 曲线拟合
    ff_l = fitting(simple_points_xyl, view=0)

    # 谷底算法
    all_min_point_x, all_min_point_z = gudi2(simple_points_xyl, ff_l, view=0)
    # 最近邻
    p_index, points_nbrs = find_nbrs_2d(np_left_2d, all_min_point_x, all_min_point_z, num=50, view=0)
    return p_index, points_nbrs
