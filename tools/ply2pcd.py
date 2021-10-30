import os
import open3d as o3d
from filter import remove_invalid

data_path = "E:/data"
input_path = os.path.join(data_path, 'input')
output_path = os.path.join(data_path, 'output')
root_ = os.path.dirname(os.path.abspath(__file__))
assert os.path.exists(input_path)

if not os.path.exists(output_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(output_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

input_filenames = os.listdir(input_path)
for filename in input_filenames:
    print(filename)
    o3d_data = o3d.io.read_point_cloud(os.path.join(input_path, filename))
    o3d_points = remove_invalid(o3d_data, view=0)
    print(o3d_points)

    output_filename = os.path.join(output_path, filename.split('.')[-2]+'.pcd')
    o3d.io.write_point_cloud(output_filename, o3d_points)

print('\n')
print('successful')


