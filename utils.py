import argparse
# import os
import numpy as np
import pandas as pd
import open3d as o3d


# 冒泡排序
from visualization import open3d_visualization_numpy


def bubbleSort_1(arr):
    n = len(arr)
    # tem = [0, 0, 0]
    # 遍历所有数组元素
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            if arr[j] < arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def bubbleSort_y_3(arr):
    # n = arr.shape[0]
    n = len(arr)
    tem = [0, 0, 0]
    # 遍历所有数组元素
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            if arr[j, 1] < arr[j + 1, 1]:
                tem[:] = arr[j, :]
                arr[j] = arr[j + 1]
                arr[j + 1] = tem[:]
    return arr

# from pyntcloud import PyntCloud
"""# 凸包检测
   hull = ConvexHull(np_points_2d)
   # plt.plot(np_points_2d[:, 0], np_points_2d[:, 1], 'o')
   simplex = hull.vertices
   # plt.plot(np_points_2d[simplex, 0], np_points_2d[simplex, 1], 'r--')
   # plt.show()
   # 搜索顶曲线点
   x_max = max(np_points_2d[simplex, 0]) + 1
   x_min = min(np_points_2d[simplex, 0]) - 1
   y_max = max(np_points_2d[simplex, 1]) + 1
   y_min = min(np_points_2d[simplex, 1]) - 1"""

"""# 边界框
aabb = o3d_point_clouds.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)
obb = o3d_point_clouds.get_oriented_bounding_box()
obb.color = (0, 1, 0)
# print(aabb.max_bound)
# print(aabb.min_bound)
# o3d.visualization.draw_geometries([ply_o3d_value, aabb])"""

"""
        # 计算法线
        # ply_o3d_value.estimate_normals(
        #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=20))

        # o3d.visualization.draw_geometries([ply_o3d_value],
        #                                   point_show_normal=True)#法线可视化
        # print(ply_o3d_value.normals[0])  # 打印第 0 个点的法向量
        # print(np.asarray(ply_o3d_value.normals)[:10, :])  # 打印前 10 个点的法向量"""
# 画出3d点云
from matplotlib import pyplot as plt


def plt_draw(points, output_filename=None):
    """ points is a Nx3 numpy array """
    # plt.plot(points[:, 0], points[:, 1], 'o')
    # 开始绘图
    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    # 标题
    plt.title('point cloud')
    # 利用xyz的值，生成每个点的相应坐标（x,y,z）
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='.')
    # ax.axis('scaled')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # 显示
    plt.show()
    if output_filename is not None:
        plt.savefig(output_filename)


def draw_wcs(long=10,):
    point = [[0, 0, 0], [long, 0, 0], [0, long, 0], [0, 0, long]]  # 提取第一第二主成分
    line = [[0, 1], [0, 2], [0, 3]]  # 由点构线，[0,1]代表以点集中序号为0和1的点组成线
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 为不同的线添加不同颜色
    # 构造open3d中的lineset对象，用于主成分的显示
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(point),
        lines=o3d.utility.Vector2iVector(line))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set





def o3d2np(o3d_temp_points):
    np_point_n_3 = np.asarray(o3d_temp_points.points)
    return np_point_n_3


def np2o3d(np_point_n_3):
    o3d_temp_points = o3d.geometry.PointCloud()
    o3d_temp_points.points = o3d.utility.Vector3dVector(np_point_n_3)
    return o3d_temp_points


def visualization_point_cloud(org_points, type_t):
    if type_t is '2d':
        temp_points = np.zeros((org_points.shape[0], 3))
        temp_points[:, 0] = org_points[:, 0]
        temp_points[:, 1] = org_points[:, 1]

        open3d_visualization_numpy(temp_points)
    elif type_t is '3d':
        open3d_visualization_numpy(org_points)





def get_voxel_grid_classifier(points, leaf_size):
    """ Get a function for 3D point -- voxel grid assignment
    Parameters
    ----------
    leaf_size:voxel尺寸
    points: (pandas.DataFrame)points in the point cloud输入点云

    """
    # get bounding box:
    (p_min, p_max) = (points.min(), points.max())
    (D_x, D_y, D_z) = (
        np.ceil((p_max['x'] - p_min['x']) / leaf_size).astype(np.int_),
        np.ceil((p_max['y'] - p_min['y']) / leaf_size).astype(np.int_),
        np.ceil((p_max['z'] - p_min['z']) / leaf_size).astype(np.int_),
    )

    def classifier(x, y, z):
        """ assign given 3D point to voxel grid
        Parameters:
            x(float): X
            y(float): Y
            z(float): Z

        Return:
            idx(int): voxel grid index
        """
        (i_x, i_y, i_z) = (
            np.floor((x - p_min['x']) / leaf_size).astype(np.int_),
            np.floor((y - p_min['y']) / leaf_size).astype(np.int_),
            np.floor((z - p_min['z']) / leaf_size).astype(np.int_),
        )
        idx = i_x + D_x * i_y + D_x * D_y * i_z
        return idx

    return classifier


def voxel_filter(points, leaf_size, method='centroid'):
    """ Downsample point cloud using voxel grid
        voxel滤波
    Parameters:
        points(pandas.DataFrame): points in the point cloud
        leaf_size(float): voxel grid resolution
        method(str): downsample method. 'centroid' or 'random'. defaults to 'centroid'

    Returns:
        filtered_points(numpy.ndarray): downsampled point cloud
    """
    filtered_points = None

    # TODO_ 03: voxel grid filtering
    working_points = points.copy(deep=True)

    # get voxel grid classifier:
    classifier = get_voxel_grid_classifier(working_points, leaf_size)
    # assign to voxel grid:
    working_points['voxel_grid_id'] = working_points.apply(
        lambda row: classifier(row['x'], row['y'], row['z']), axis=1
    )

    # centroid:
    if method == 'centroid':
        filtered_points = working_points.groupby(['voxel_grid_id']).mean().to_numpy()
    elif method == 'random':
        filtered_points = working_points.groupby(['voxel_grid_id']).apply(
            lambda x: x[['x', 'y', 'z']].sample(1)
        ).to_numpy()

    return filtered_points


def get_arguments():
    """ Get command-line arguments
    """
    # init parser:
    parser = argparse.ArgumentParser(description="Downsample given point cloud using voxel grid.")

    # add required and optional groups:
    parser.add_argument("-p", '--point_cloud', default='1point1010.ply', type=str, nargs='+',
                        help='the file name of point_cloud ')

    # parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], required=True,
    #                     help="increase output verbosity")

    # parse arguments:解析
    arguments = parser.parse_args()

    return arguments.point_cloud


class PCAa:
    """ pca = PCA()
     pca.fit(data, 2)
     data_pca = pca.transform(data)
     print(pca.val_)
     print(pca.vec_)
     """

    def __init__(self):
        self.val_ = []
        self.vec_ = np.array([])

    def fit(self, df, k):
        if k > df.shape[1]:
            print('k must lower than feature number')
        else:
            df_scale = (df - df.mean()) / df.std()  # z-score标准化
            df_cov = np.cov(df_scale.T)  # 协方差矩阵
            val, vec = np.linalg.eig(df_cov)  # 协方差矩阵特征值、特征向量
            index = np.argsort(-val)[:k]  # 求出特征向量从大到小排列的索引
            val = val[index]  # 特征值从大到小重新排列
            vec = vec[:, index]  # 特征值相应的特征向量也重新排列
            self.val_, self.vec_ = val, vec

    def transform(self, df):
        col_names = df.columns
        final = np.dot(df - df.mean(), self.vec_.T)  # m*n维的原始数据矩阵与n*k特征向量矩阵点乘即为降维后的结果
        return pd.DataFrame(final, columns=[str(i) + '_pca' for i in col_names])


def select_point():
    # Load data
    pcd = o3d.io.read_point_cloud("../../test_data/fragment.ply")
    vol = o3d.visualization.read_selection_polygon_volume(
        "../../test_data/Crop/cropped.json")
    chair = vol.crop_point_cloud(pcd)




if __name__ == '__main__':
    # 我们先随机生成一些数字,作为点云输入,为了减少物体尺度的问题,
    # 通常会将点云缩到半径为1的球体中
    # 为了方便起见,LZ把batch_size改成1
    point_cloud = np.random.rand(1024, 3)
    plt_draw(point_cloud)
