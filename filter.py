import time
import numpy as np
import open3d as o3d

import utils

from visualization import display_inlier_outlier3, display_inlier_outlier


def Pretreatment(file_name):
    # 点云预处理
    # 1.读取文件
    o3d_data = o3d.io.read_point_cloud(file_name)  # 0.034793
    # o3d.visualization.draw_geometries([o3d_data])

    # 2.去除无效点
    o3d_points = remove_invalid(o3d_data, view=0)  # 0.073476
    # TODO: 1.去掉append

    # 3.下采样
    o3d_points_ds = down_simple(o3d_points, view=0)
    # NOTE:体素下采样可以解决由于密度原因造成的主成分向量偏移
    # TODO: 2.实验重做
    # TODO: 3.最远点采样

    # 4.孤点滤波
    o3d_points_ds_sf = Soliton_filter(o3d_points_ds, view=0)

    # 5.点云分割
    o3d_under_points = division(o3d_points_ds_sf, view=0)
    # o3d_under_points1,o3d_point_clouds2, = separate(o3d_under_points)
    # TODO: 4.分离

    return o3d_under_points


def cur_down_simple(o3d_data):
    # 曲率下采样
    def vector_angle(x, y):
        Lx = np.sqrt(x.dot(x))
        Ly = (np.sum(y ** 2, axis=1)) ** 0.5
        cos_angle = np.sum(x * y, axis=1) / (Lx * Ly)
        angle = np.arccos(cos_angle)
        angle2 = angle * 360 / 2 / np.pi
        return angle2

    knn_num = 30  # 自定义参数值(邻域点数)
    angle_thre = 30  # 自定义参数值(角度值)
    N = 5  # 自定义参数值(每N个点采样一次)
    C = 20  # 自定义参数值(采样均匀性>N)

    # pcd = o3d.io.read_point_cloud(path)
    point = np.asarray(o3d_data.points)
    point_size = point.shape[0]
    tree = o3d.geometry.KDTreeFlann(o3d_data)
    o3d.geometry.PointCloud.estimate_normals(
        o3d_data, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_num))
    normal = np.asarray(o3d_data.normals)
    normal_angle = np.zeros(point_size)
    for i in range(point_size):
        [_, idx, _] = tree.search_knn_vector_3d(point[i], knn_num + 1)
        current_normal = normal[i]
        knn_normal = normal[idx[1:]]
        normal_angle[i] = np.mean(vector_angle(current_normal, knn_normal))

    point_high = point[np.where(normal_angle >= angle_thre)]
    point_low = point[np.where(normal_angle < angle_thre)]
    pcd_high = o3d.geometry.PointCloud()
    pcd_high.points = o3d.utility.Vector3dVector(point_high)
    pcd_low = o3d.geometry.PointCloud()
    pcd_low.points = o3d.utility.Vector3dVector(point_low)
    pcd_high_down = o3d.geometry.PointCloud.uniform_down_sample(pcd_high, N)
    pcd_low_down = o3d.geometry.PointCloud.uniform_down_sample(pcd_low, C)
    pcd_finl = o3d.geometry.PointCloud()
    pcd_finl.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd_high_down.points),
                                                                 np.asarray(pcd_low_down.points))))
    print(pcd_finl)
    display_inlier_outlier3(pcd_high_down, pcd_low_down)
    # o3d.visualization.draw_geometries([pcd_finl])


def down_simple(o3d_data, view=0):
    # time.sleep(1)
    a = time.perf_counter()

    # 1.均匀下采样:一排一排去除，不太好，但速度快
    # data_new = o3d_data.uniform_down_sample(every_k_points=20)
    """参数实验
        2 : 128000 points / 0.003708 s
        3 : 85334 points / 0.002520 s
        4 : 64000 points / 0.001674 s
        5 : 51200 points / 0.001425 s
        6 : 42667 points / 0.001412 s
        7 : 36572 points / 0.001349 s
        8 : 32000 points / 0.001228 s
        9 : 28445 points / 0.001178 s
        10 : 25600 points / 0.001211 s
        11 : 23273 points / 0.001305 s
        12 : 21334 points / 0.001172 s
        13 : 19693 points / 0.001061 s
        14 : 18286 points / 0.001013 s
        15 : 17067 points / 0.000975 s
        16 : 16000 points / 0.001077 s
        17 : 15059 points / 0.001013 s
        18 : 14223 points / 0.000778 s
        19 : 13474 points / 0.000742 s
        20 : 12800 points / 0.000795 s
        21 : 12191 points / 0.000714 s
        22 : 11637 points / 0.000693 s
        23 : 11131 points / 0.000672 s
        24 : 10667 points / 0.000628 s
        25 : 10240 points / 0.000650 s
        26 :  9847 points / 0.000694 s
        27 :  9482 points / 0.000681 s
        28 :  9143 points / 0.000670 s
        29 :  8828 points / 0.000623 s
        30 :  8534 points / 0.000592 s
        31 :  8259 points / 0.000586 s
        32 :  8000 points / 0.000529 s
        33 :  7758 points / 0.000537 s
        34 :  7530 points / 0.000522 s
        35 :  7315 points / 0.000523 s
        36 :  7112 points / 0.000533 s
        37 :  6919 points / 0.000540 s
        38 :  6737 points / 0.000505 s
        39 :  6565 points / 0.000532 s
        40 :  6400 points / 0.000480 s
        41 :  6244 points / 0.000527 s
        42 :  6096 points / 0.000488 s
        43 :  5954 points / 0.000497 s
        44 :  5819 points / 0.000503 s
        45 :  5689 points / 0.000486 s
        46 :  5566 points / 0.000486 s
        47 :  5447 points / 0.000567 s
        48 :  5334 points / 0.000468 s
        49 :  5225 points / 0.000482 s
    """
    # 2.体素下采样，慢点
    data_new = o3d_data.voxel_down_sample(voxel_size=0.8)
    """参数实验
        0.5 : 49396 points / 0.014443 s
        0.6 : 38592 points / 0.012225 s
        0.7 : 31043 points / 0.011160 s
        0.8 : 25261 points / 0.010560 s
        0.9 : 21221 points / 0.009168 s
        1.0 : 17950 points / 0.009191 s
        1.1 : 15516 points / 0.008631 s
        1.2 : 13523 points / 0.008365 s
        1.3 : 11682 points / 0.007945 s
        1.4 : 10394 points / 0.007558 s
        1.5 : 9331 points  / 0.007409 s
        1.6 : 8375 points  / 0.007708 s
        1.7 : 7592 points  / 0.007528 s
        1.8 : 6985 points  / 0.007406 s
        1.9 : 6299 points  / 0.006916 s
        2.0 : 5734 points  / 0.007894 s
        3.0 : 2872 points  / 0.006973 s
        4.0 : 1687 points  / 0.006394 s
        5.0 : 1145 points  / 0.006295 s
        5.9 : 823 points   / 0.006878 s
       """
    # ply_new = o3d.geometry.PointCloud.voxel_down_sample_and_trace(ply, 3 ,)

    # 3.曲率下采样 太慢
    # data_new = cur_down_simple(o3d_data)

    # 4.随机下采样
    # data_new = o3d.geometry.PointCloud.random_down_sample(o3d_data, sampling_ratio=0.1)
    """参数实验
        0.01 :  2560 points / 0.005108 s
        0.02 :  5120 points / 0.005234 s
        0.03 :  7680 points / 0.005364 s
        0.04 : 10240 points / 0.005497 s
        0.05 : 12800 points / 0.005531 s
        0.06 : 15360 points / 0.005698 s
        0.07 : 17919 points / 0.005836 s
        0.08 : 20480 points / 0.005810 s
        0.09 : 23040 points / 0.005791 s
    """

    # # 自动输出
    # for i in np.arange(0.01, 0.1, 0.01):
    #     a = time.perf_counter()
    #     data_new = o3d_data.random_down_sample(i)
    #     b = time.perf_counter()
    #     data_new = utils.o3d2np(data_new)
    #     print("%.2f : %5d points / %.6f s" % (i, data_new.shape[0], (b - a)))

    if view:
        b = time.perf_counter()
        print("%.6f" % (b - a))
        print(data_new)
        o3d.visualization.draw_geometries([data_new])

    return data_new


def remove_invalid(o3d_data, view=0):
    # 去除无效点
    points = []
    tem_point = utils.o3d2np(o3d_data)
    for i in range(tem_point.shape[0]):
        if tem_point[i][0]:
            points.append(tem_point[i])
        # else:
        #     pass
            # print(tem_point[i])
    points = np.array(points)
    o3d_points = utils.np2o3d(points)
    if view:
        print(o3d_points)
        o3d.visualization.draw_geometries([o3d_points])
    return o3d_points


def Soliton_filter(o3d_points, view=0):
    # 孤点滤波
    a = time.perf_counter()
    # 统计滤波
    cl, ind = o3d.geometry.PointCloud.remove_statistical_outlier(o3d_points, nb_neighbors=50, std_ratio=1.0)  # 0.002564
    # TODO: 计算平均距离画曲线，自动提取参数
    # 半径滤波
    # cl, ind = o3d_points.remove_radius_outlier(nb_points=10, radius=5)  # 0.007242

    if view:
        b = time.perf_counter()
        print("%.6f" % (b - a))
        display_inlier_outlier(o3d_points, ind)  # 显示的原始点不是cl
        o3d.visualization.draw_geometries([cl])
        print(cl)
    return cl


def division(o3d_points_ds_sf, view=0, method='select'):
    a = time.perf_counter()
    # 分割

    # 归一化
    np_points_xyz = utils.o3d2np(o3d_points_ds_sf)
    from my_pca import normalization
    np_points_normalized, mo = normalization(np_points_xyz)

    # 剪裁点云
    jjj = []
    if method == 'select':  # 0.002202s
        for i in range(np_points_normalized.shape[0]):
            if np_points_normalized[i, 2] > 0:
                jjj.append(i)
        o3d_under_points = o3d_points_ds_sf.select_by_index(jjj)
    elif method == 'append':  # 0.0034320 s
        for i in range(np_points_normalized.shape[0]):
            if np_points_normalized[i, 2] > 0:
                jjj.append(np_points_normalized[i])
        o3d_under_points = utils.np2o3d(np.array(jjj) + mo)
    else:
        o3d_under_points = 0  # 强制出错

    if view:
        b = time.perf_counter()
        print("%.6f" % (b - a))
        o3d_under_points.paint_uniform_color([1, 0, 0])
        o3d_points_ds_sf.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([o3d_points_ds_sf, o3d_under_points])
    return o3d_under_points


if __name__ == '__main__':
    pass
