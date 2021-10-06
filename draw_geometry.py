import open3d as o3d


def Line_Set(point, line, colors):
    line_set = o3d.geometry.LineSet(  # 构造open3d中的lineset对象，用于主成分的显示
        points=o3d.utility.Vector3dVector(point),
        lines=o3d.utility.Vector2iVector(line))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def draw_axis(u=None, long=10):
    if u is not None:
        # point = [[0, 0, 0], u[:, 0] * -long, u[:, 1] * -long, u[:, 2] * long]  # 提取第一第二主成分
        point = [[0, 0, 0], u[:, 0] * long, u[:, 1] * long, u[:, 2] * long]
        line = [[0, 1], [0, 2], [0, 3]]  # 由点构线，[0,1]代表以点集中序号为0和1的点组成线
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 为不同的线添加不同颜色
        line_set = Line_Set(point, line, colors)
    else:
        point = [[0, 0, 0], [long, 0, 0], [0, long, 0], [0, 0, long]]  # 提取第一第二主成分
        line = [[0, 1], [0, 2], [0, 3]]  # 由点构线，[0,1]代表以点集中序号为0和1的点组成线
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 为不同的线添加不同颜色
        # 构造open3d中的lineset对象，用于主成分的显示
        line_set = Line_Set(point, line, colors)
    return line_set


def draw():
    print("绘制立方体")
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0,  # x长度
                                                    height=1.0,  # y长度
                                                    depth=1.0)  # z长度
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    o3d.visualization.draw_geometries([mesh_box])

    # 绘制箭头
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=1.0,  # 圆柱体的半径
                                                   cone_radius=1.5,  # 圆锥的半径
                                                   cylinder_height=5.0,  # 圆柱体的高度。圆柱体从(0,0,0)到(0,0，圆筒高度)
                                                   cone_height=4.0,  # 圆锥的高度。圆锥的轴将从(0,0，圆筒高度)到(0,0，圆筒高度+圆锥高度)
                                                   resolution=20,  # 圆锥将被分割成“分辨率”段。
                                                   cylinder_split=4,  # "圆柱高度"将被分割成"圆柱分割"段
                                                   cone_split=1)  # “cone_height”将被分割成“cone_split”段。
    arrow.compute_vertex_normals()
    arrow.paint_uniform_color([1, 0, 0])
    print("绘制箭头")
    o3d.visualization.draw_geometries([arrow])

    # 绘制球体

    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0,  # 球的半径
                                                          resolution=100)  # 图形显示的分辨率，可省略。默认值为：20
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
    o3d.visualization.draw_geometries([mesh_sphere])

    # 绘制圆柱
    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.3,  # 半径
                                                              height=4.0)  # 高(有第三个参数为分辨率)
    mesh_cylinder.compute_vertex_normals()
    mesh_cylinder.paint_uniform_color([0.1, 0.4, 0.1])
    o3d.visualization.draw_geometries([mesh_cylinder])

    # 绘制箭头
    cone = o3d.geometry.TriangleMesh.create_cone(radius=1.0,  # 圆锥的半径
                                                 height=2.0,  # 圆锥的高度
                                                 resolution=20,
                                                 split=1)  # The ``height`` will be split into ``split`` segments

    cone.compute_vertex_normals()
    cone.paint_uniform_color([0, 1, 0])
    print("绘制箭头")
    o3d.visualization.draw_geometries([cone])


if __name__ == "__main__":
    draw()
