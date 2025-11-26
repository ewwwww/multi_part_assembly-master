import os
import random

import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

from torch.utils.data import Dataset, DataLoader


class GeometryPartDataset(Dataset):
    """Geometry part assembly dataset.

    We follow the data prepared by Breaking Bad dataset:
        https://breaking-bad-dataset.github.io/
    """

    def __init__(
        self,
        data_dir,
        data_fn,# 数据列表文件名
        data_keys,# 额外加载的数据，如 ('part_ids', 'instance_label')
        category='',# 数据类别，'all' 表示所有类别，否则只使用指定类别
        num_points=1000,# 每个部件的点数
        min_num_part=2,# 最小部件数量
        max_num_part=20,# 最大部件数量，超过这个数量的样本将被忽略
        shuffle_parts=False,# 是否打乱部件顺序
        rot_range=-1,# 旋转范围-1 表示不旋转，正值表示旋转范围（以度为单位），用于 Curriculum Learning
        overfit=-1,# 过拟合测试模式-1 表示使用全部数据，正值表示只使用前 N 个样本（用于快速调试）
    ):
        # store parameters
        self.category = category if category.lower() != 'all' else ''# 数据类别，'all' 表示所有类别，否则只使用指定类别
        self.data_dir = data_dir
        self.num_points = num_points
        self.min_num_part = min_num_part
        self.max_num_part = max_num_part  # ignore shapes with more parts
        self.shuffle_parts = shuffle_parts  # shuffle part orders
        self.rot_range = rot_range  # rotation range in degree

        # list of fracture folder path
        self.data_list = self._read_data(data_fn)
        if overfit > 0:# 过拟合测试模式-1 表示使用全部数据，正值表示只使用前 N 个样本（用于快速调试）
            self.data_list = self.data_list[:overfit]

        # additional data to load, e.g. ('part_ids', 'instance_label')
        self.data_keys = data_keys

    def _read_data(self, data_fn):#读取数据：从数据列表文件中读取数据，并过滤掉无效的部件数量
        """Filter out invalid number of parts."""
        with open(os.path.join(self.data_dir, data_fn), 'r') as f:# 把目录和文件名合成一个路径(mesh_list:规范化路径格式，即确保路径是跨平台的)
            mesh_list = [line.strip() for line in f.readlines()]# 去除开头结尾空格，将文件路径存入mesh_list
            if self.category:
                mesh_list = [
                    line for line in mesh_list
                    if self.category in line.split('/')#将路径按'/'分割成列表，检查是否包含指定类别
                ]#例如：'plate/1d4093ad2dfad9df24be2e4f911ee4af' → ['plate', '1d4093ad2dfad9df24be2e4f911ee4af']
        data_list = []
        for mesh in mesh_list:
            # 将路径和mesh拼接，得到mesh_dir: xxx/plate/1d4093ad2dfad9df24be2e4f911ee4af/fractured_0
            mesh_dir = os.path.join(self.data_dir, mesh)
            if not os.path.isdir(mesh_dir):
                print(f'{mesh} does not exist')
                continue
            for frac in os.listdir(mesh_dir):#os.listdir(mesh_dir)：返回目录下的所有文件和文件夹名，例如：['fractured_0', 'fractured_1', 'mode_0', 'other_folder']
                # we take both fractures and modes for training,过滤掉非fractured和mode的文件夹 
                if 'fractured' not in frac and 'mode' not in frac:#如果文件夹名不包含'fractured'和'mode'，则跳过
                    continue
                frac = os.path.join(mesh, frac)#将路径和文件夹名拼接，得到frac_dir: xxx/plate/1d4093ad2dfad9df24be2e4f911ee4af/fractured_0
                num_parts = len(os.listdir(os.path.join(self.data_dir, frac)))#计算部件数量，构建 fracture 的完整路径，统计该文件夹中的文件数量（每个文件对应一个部件）
                #例如：如果 fractured_0 中有 part_0.obj, part_1.obj, part_2.obj，则 num_parts = 3
                if self.min_num_part <= num_parts <= self.max_num_part:#如果部件数量在最小和最大数量之间，则添加到data_list
                    data_list.append(frac)
        return data_list

    @staticmethod
    def _recenter_pc(pc):#点云中心化：将点云平移到原点，使点云的质心位于 (0, 0, 0)
        """pc: [N, 3]"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid[None]
        return pc, centroid

    def _rotate_pc(self, pc):#部件点云旋转：将点云绕某个轴随机旋转，使点云的朝向随机
        """pc: [N, 3]"""
        if self.rot_range > 0.:
            rot_euler = (np.random.rand(3) - 0.5) * 2. * self.rot_range# 在[-rot_range, rot_range]范围内随机生成3个角度
            rot_mat = R.from_euler('xyz', rot_euler, degrees=True).as_matrix()#将角度转换为旋转矩阵
        else:
            rot_mat = R.random().as_matrix()
        pc = (rot_mat @ pc.T).T#将点云旋转
        quat_gt = R.from_matrix(rot_mat.T).as_quat()#将旋转矩阵转换为四元数
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        return pc, quat_gt

    @staticmethod
    def _shuffle_pc(pc):#部件点云打乱：将点云的顺序随机打乱
        """pc: [N, 3]"""
        order = np.arange(pc.shape[0])#生成0到N-1的数组
        random.shuffle(order)#将数组随机打乱
        pc = pc[order]#将点云按照打乱后的顺序重新排列
        return pc

    def _pad_data(self, data):#填充数据：将数据填充到指定形状，用于后续计算损失
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...]."""
        data = np.array(data)
        pad_shape = (self.max_num_part, ) + tuple(data.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[:data.shape[0]] = data
        return pad_data

    def _get_pcs(self, data_folder):# 从 fracture 文件夹读取所有 mesh 文件，将每个 mesh 采样为点云，返回所有部件的点云数组
        """Read mesh and sample point cloud from a folder."""
        # `data_folder`: xxx/plate/1d4093ad2dfad9df24be2e4f911ee4af/fractured_0
        data_folder = os.path.join(self.data_dir, data_folder)
        mesh_files = os.listdir(data_folder)
        mesh_files.sort()# 拼接完整路径；排序
        if not self.min_num_part <= len(mesh_files) <= self.max_num_part:#如果部件数量不在最小和最大数量之间，则抛出错误
            raise ValueError

        # shuffle part orders
        if self.shuffle_parts:# 如果需要打乱部件顺序，则打乱部件顺序
            random.shuffle(mesh_files)

        # read mesh and sample points
        meshes = [
            trimesh.load(os.path.join(data_folder, mesh_file))# 加载mesh文件
            for mesh_file in mesh_files
        ]
        pcs = [
            trimesh.sample.sample_surface(mesh, self.num_points)[0]# 采样点云
            for mesh in meshes
        ]
        return np.stack(pcs, axis=0)# 将所有部件的点云数组堆叠成一个数组

    def __getitem__(self, index):# 获取数据：返回一个字典，包含所有部件的点云、旋转和平移信息
        pcs = self._get_pcs(self.data_list[index])
        num_parts = pcs.shape[0]
        cur_pts, cur_quat, cur_trans = [], [], []
        for i in range(num_parts):
            pc = pcs[i]
            pc, gt_trans = self._recenter_pc(pc)
            pc, gt_quat = self._rotate_pc(pc)
            cur_pts.append(self._shuffle_pc(pc))# 打乱部件碎片的排列顺序
            cur_quat.append(gt_quat)#旋转（四元数）
            cur_trans.append(gt_trans)#平移后原点(0,0,0)
        cur_pts = self._pad_data(np.stack(cur_pts, axis=0))  # [P, N, 3]
        cur_quat = self._pad_data(np.stack(cur_quat, axis=0))  # [P, 4] 
        cur_trans = self._pad_data(np.stack(cur_trans, axis=0))  # [P, 3] 
        """
        data_dict = {
            'part_pcs': MAX_NUM x N x 3
                The points sampled from each part.

            'part_trans': MAX_NUM x 3
                Translation vector

            'part_quat': MAX_NUM x 4
                Rotation as quaternion.

            'part_valids': MAX_NUM
                1 for shape parts, 0 for padded zeros.

            'instance_label': MAX_NUM x 0, useless

            'part_label': MAX_NUM x 0, useless

            'part_ids': MAX_NUM, useless

            'data_id': int
                ID of the data.

        }
        """

        data_dict = {
            'part_pcs': cur_pts,# [MAX_NUM, N, 3] 每个部件的点云（打乱后的）
            'part_quat': cur_quat,# [MAX_NUM, 4] 旋转（四元数）
            'part_trans': cur_trans,# [MAX_NUM, 3] 平移后原点(0,0,0)
        }
        # valid part masks 创建二进制掩码数组，标记哪些位置是真实部件（1），哪些是填充的零（0），用于后续计算损失
        valids = np.zeros((self.max_num_part), dtype=np.float32)# 创建长度为 max_num_part 的全零数组，用于标记有效部件
        valids[:num_parts] = 1.# 将前 num_parts 位置设置为 1，表示这些位置是有效部件
        data_dict['part_valids'] = valids# [MAX_NUM] 有效部件掩码
        # data_id
        data_dict['data_id'] = index# 数据索引
        # instance_label is useless in non-semantic assembly
        # keep here for compatibility with semantic assembly
        # make its last dim 0 so that we concat nothing
        instance_label = np.zeros((self.max_num_part, 0), dtype=np.float32)# [MAX_NUM, 0] 无效部件掩码
        data_dict['instance_label'] = instance_label# [MAX_NUM, 0] 无效部件掩码
        # the same goes to part_label
        part_label = np.zeros((self.max_num_part, 0), dtype=np.float32)# [MAX_NUM, 0] 无效部件掩码
        data_dict['part_label'] = part_label# [MAX_NUM, 0] 无效部件掩码

        for key in self.data_keys:
            if key == 'part_ids':
                cur_part_ids = np.arange(num_parts)  # p
                data_dict['part_ids'] = self._pad_data(cur_part_ids)

            elif key == 'valid_matrix':
                out = np.zeros((self.max_num_part, self.max_num_part),
                               dtype=np.float32)
                out[:num_parts, :num_parts] = 1.
                data_dict['valid_matrix'] = out

            else:
                raise ValueError(f'ERROR: unknown data {key}')

        return data_dict

    def __len__(self):
        return len(self.data_list)


def build_geometry_dataloader(cfg):# 构建数据加载器：返回训练和验证数据加载器
    data_dict = dict(
        data_dir=cfg.data.data_dir,
        data_fn=cfg.data.data_fn.format('train'),
        data_keys=cfg.data.data_keys,# data_keys 是必需的参数，用于控制可选数据项，如 ('part_ids', 'instance_label')
        category=cfg.data.category,
        num_points=cfg.data.num_pc_points,
        min_num_part=cfg.data.min_num_part,
        max_num_part=cfg.data.max_num_part,
        shuffle_parts=cfg.data.shuffle_parts,
        rot_range=cfg.data.rot_range,
        overfit=cfg.data.overfit,
    )
    train_set = GeometryPartDataset(**data_dict)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.exp.batch_size,
        shuffle=True,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.exp.num_workers > 0),
    )

    data_dict['data_fn'] = cfg.data.data_fn.format('val')
    data_dict['shuffle_parts'] = False
    val_set = GeometryPartDataset(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.exp.batch_size * 2,
        shuffle=False,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.exp.num_workers > 0),
    )
    return train_loader, val_loader
