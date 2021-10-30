"""
===================================
Demo of POD, by le
===================================
"""
# import numpy as np
from Deal2d import *
from Feature_processing import *
from filter import *
# from visualization import custom_draw_geometry_with_rotation2

print(__doc__)


def main():
    a = time.perf_counter()
    # parse arguments:
    file_name = utils.get_arguments()  # 0.000424

    # 点云预处理
    o3d_point_cloud = Pretreatment(file_name)

    # 姿态矫正
    np_point_clouds_rotate = pos_cor(o3d_point_cloud)

    # 特征分割
    np_pc_left, np_pc_right = separate(np_point_clouds_rotate, view=0)

    # 参数投影
    np_left_2d, np_right_2d = projection(np_pc_left, np_pc_right)

    # batch采样处理
    index_left, points_nbrs2d_left = simple_deal(np_left_2d)
    index_right, points_nbrs2d_right = simple_deal(np_right_2d)

    # 插值映射
    points_nbrs3d_left = interpolation_mapping(index_left, np_pc_left, view=0)
    points_nbrs3d_right = interpolation_mapping(index_right, np_pc_right, view=0)

    # 取区域中心
    xyz5_l = [np.mean(points_nbrs3d_left[i][0], axis=0) for i in range(len(points_nbrs3d_left))]
    xyz5_r = [np.mean(points_nbrs3d_right[i][0], axis=0) for i in range(len(points_nbrs3d_right))]

    # 中点

    xx = [np.mean((xyz5_l[i], xyz5_l[i + 1]), axis=0) for i in range(len(xyz5_l) - 1)]

    # 关节突可视化
    point1 = np.squeeze(np.concatenate(points_nbrs3d_left, 1))
    point2 = np.squeeze(np.concatenate(points_nbrs3d_right, 1))
    point = np.squeeze(np.concatenate([point2, point1], 0))
    display_inlier_outlier2(np_point_clouds_rotate, point-0.1)

    point = np.array(xx)

    point[:, 1] = point[:, 1] + 4

    _, _ = find_nbrs_3d(np_point_clouds_rotate, point, radius=3, view=0)

    # display_inlier_outlier2(np_point_clouds_rotate, point)
    # object_cut_points = utils.np2o3d(np_point_clouds_rotate)
    # mid_points = utils.np2o3d(point)
    #
    # object_cut_points.paint_uniform_color([0.8, 0.8, 0.8])
    # mid_points.paint_uniform_color([1, 0, 0])
    # custom_draw_geometry_with_rotation2(object_cut_points, mid_points)

    # 重检测

    b = time.perf_counter()
    print("代码运行时长：%.3fs" % (b - a))


if __name__ == '__main__':
    main()
