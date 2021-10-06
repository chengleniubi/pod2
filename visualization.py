import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt


def open3d_visualization_numpy(org_points):
    o3d_temp_points = o3d.geometry.PointCloud()
    o3d_temp_points.points = o3d.utility.Vector3dVector(org_points)
    o3d.visualization.draw_geometries([o3d_temp_points])


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    # print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def display_inlier_outlier2(cloud, points):
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(cloud)
    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(points)

    # print("Showing outliers (red) and inliers (gray): ")
    o3d_points.paint_uniform_color([1, 0, 0])
    o3d_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # o3d_points.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([o3d_cloud, o3d_points])


def display_inlier_outlier3(o3d_cloud, o3d_points):
    # print("Showing outliers (red) and inliers (gray): ")
    o3d_points.paint_uniform_color([1, 0, 0])
    o3d_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([o3d_cloud, o3d_points])


def view_two_cloud(source, target):
    # 可视化多个点云
    source.paint_uniform_color([1, 0, 0])
    target.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([source, target])


def random_color(source):
    # 可视化随机颜色的点云 效果不明显
    source.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(1, 3)))
    o3d.visualization.draw_geometries([source])


def viss(data, label):
    """可视化点云加标签
    :param data: n*3的矩阵
    :param label: n*1的矩阵
    :return: 可视化
    """
    data = data[:, :3]
    labels = np.asarray(label)
    max_label = max(labels)

    # 颜色
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))

    pt1 = o3d.geometry.PointCloud()
    pt1.points = o3d.utility.Vector3dVector(data.reshape(-1, 3))
    pt1.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pt1], 'part of cloud', width=500, height=500)


# 方法2 类
# 参考：https://github.com/Jiang-Muyun/Open3D-Semantic-KITTI-Vis/blob/ddb188e1375a1d464dec077826725afd72a85e63/src/kitti_base.py#L43
class PtVis():
    def __init__(self, name='20m test', width=800, height=600, json='./config/view_point.json'):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=name, width=width, height=height)
        self.axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])

        # 可视化参数
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 1
        opt.show_coordinate_frame = True

        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        # 读取viewpoint参数
        self.param = o3d.io.read_pinhole_camera_parameters(json)
        self.ctr = self.vis.get_view_control()
        self.ctr.convert_from_pinhole_camera_parameters(self.param)
        print('viewpoint json loaded!')

        # param = self.ctr.convert_to_pinhole_camera_parameters()
        #
        # o3d.io.write_pinhole_camera_parameters('./config/new.json',param)
        # print('viewpoint json saved!')

    def __del__(self):
        self.vis.destroy_window()

    def update(self, pcd):
        """
        :param pcd: PointCLoud()
        :return:
        """
        self.pcd.points = pcd.points
        self.pcd = pcd

        # self.pcd.colors=pcd.colors

        # self.vis.clear_geometries()
        self.vis.update_geometry(self.pcd)  # 更新点云

        # self.vis.remove_geometry(self.pcd)          # 删除vis中的点云
        self.vis.add_geometry(self.pcd)  # 增加vis中的点云

        # 设置viewpoint
        self.ctr.convert_from_pinhole_camera_parameters(self.param)

        self.vis.poll_events()
        self.vis.update_renderer()
        # self.vis.run()

    def capture_screen(self, fn, depth=False):
        if depth:
            self.vis.capture_depth_image(fn, False)
        else:
            self.vis.capture_screen_image(fn, False)


def custom_draw_geometry_with_custom_fov(points, fov_step):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(points)
    ctr = vis.get_view_control()
    print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    ctr.change_field_of_view(step=fov_step)
    print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
    vis.run()
    vis.destroy_window()


def custom_draw_geometry_with_rotation(pcd):
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(2.0, 2.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd], rotate_view)


def custom_draw_geometry_with_rotation2(pcd, pcd2):
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(1, 1.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd2, pcd], rotate_view)


def custom_draw_geometry_with_key_callback(pcd):
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "../../TestData/renderoption.json")
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)


if __name__ == "__main__":
    filename = "1point1010.ply"
    ply = o3d.io.read_point_cloud(filename)
    # random_color(ply)
    custom_draw_geometry_with_rotation(ply)
