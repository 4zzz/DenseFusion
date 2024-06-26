import torch
import glob
from pathlib import Path
import os
import cv2
import numpy as np
import numpy.ma as ma
import torchvision.transforms as transforms
from PIL import Image
from plyfile import PlyData, PlyElement
from matplotlib import pyplot as plt
import json


def get_transforms(transform):
    """
    Fixes bad transform from det(R) = -1 to det(R) = 1
    :param transform: original transform
    :return: original transform, fixed transformation
    """
    orig_transform = transform
    proper_transform = np.copy(transform)
    if np.linalg.det(orig_transform) < 0:
        proper_transform[:, 1] *= -1

    return orig_transform, proper_transform


def transform_points(points, tranformation_4x4):
    p = np.column_stack((points, np.ones(points.shape[0])))
    # print(tranformation_4x4)
    tp = (tranformation_4x4 @ p.T).T
    d = np.atleast_2d(tp[:, 3]).T
    n = tp  # (tp / d)
    return n[:, :3]


def save_ply(xyz, file, text=False):
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    xyz = xyz.view(dtype=np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')]))
    xyz = xyz.reshape(xyz.shape[:-1])
    vertex_element = PlyElement.describe(xyz, 'vertex')
    PlyData([vertex_element], text=text).write(file)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def load_json(file):
  with open(file, 'r') as openfile:
    return json.load(openfile)

class BinDataset(torch.utils.data.Dataset):

    def load_xyz(self, index):
        """
        Loads pointcloud for a given entry
        :param entry: entry from self.entries
        :return: pointcloud wit shape (3, height, width)
        """
        # exr_path = os.path.join(self.dataset_dir, entry['exr_positions_path'])
        exr_path = self.entries[index]['positions_file']
        xyz = cv2.imread(exr_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if xyz is None:
            # print(exr_path)
            raise ValueError("Image at path ", exr_path)
        # xyz = cv2.resize(xyz, (self.width, self.height), interpolation=cv2.INTER_NEAREST_EXACT)
        xyz = image_resize(xyz, width=self.width,
                           inter=cv2.INTER_NEAREST_EXACT)
        xyz = np.transpose(xyz, [1, 0, 2])
        non_zeros = np.prod(xyz, axis=2) != 0
        xyz = xyz[non_zeros]

        choose = np.random.choice(np.arange(xyz.shape[0]), self.num_points)
        return xyz[choose], choose

    def load_rgb(self, index):
        img = Image.open(self.entries[index]['image_file']).convert("RGB")
        img = np.array(img)
        img = image_resize(img, width=self.width)
        # print('img shape is', img.shape)
        img = img[:, :, :3]
        img = np.transpose(img, (2, 0, 1))
        return img

    def load_transform(self, index):
        transformation_file = self.entries[index]['transformation_file']
        t = np.loadtxt(transformation_file, max_rows=1)
        t = t.reshape((4, 4)).T
        _, proper = get_transforms(t)
        return proper

    def load_ply(self, file):
        ply_data = PlyData.read(file)
        vertex_data = ply_data['vertex']
        point_cloud = []
        for vertex in vertex_data:
            point = [vertex['x'], vertex['y'], vertex['z']]
            point_cloud.append(point)
        return np.array(point_cloud)

    def __init__(self, dataset_root, mode, num_points, width, height, preload=True, return_transform_mat=False):
        self.width = width
        self.height = height
        self.num_points = num_points
        self.model_points_num = 500
        self.num_pt = num_points
        self.isPreloaded = preload
        self.return_transform_mat = return_transform_mat

        if mode == 'train':
            entries = load_json('datasets/bin_dataset/train.json')
        elif mode == 'test':
            entries = load_json('datasets/bin_dataset/test.json')
        else:
            raise ValueError('invalid mode')

        print(f'Found {len(entries)} samples')
        # print(entries)
        self.entries = entries
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.preloaded = []
        if preload:
          for i in range(len(entries)):
              print('preloading', i+1, '/', len(entries))
              self.preloaded.append(self.preload(i))

    def __getitem__(self, index):
        return self.preloaded[index] if self.isPreloaded else self.preload(index)

    def preload(self, index):
        img = self.load_rgb(index)
        transform = self.load_transform(index)
        model_points = self.load_ply(
            self.entries[index][f'model{self.model_points_num}'])
        transform = self.load_transform(index)
        target = transform_points(model_points, transform)

        # img_numpy = np.array(img)
        mask = ma.getmaskarray(ma.masked_array(img, mask=np.ones(img.shape)))
        # print('got mask shape', mask.shape)

        xyz, choose = self.load_xyz(index)
        choose = np.array([choose])

        scale = 1.0/1000.0

        if self.return_transform_mat == False:
            return torch.from_numpy(xyz.astype(np.float32)*scale), \
                torch.LongTensor(choose.astype(np.int32)), \
                self.norm(torch.from_numpy(img.astype(np.float32))), \
                torch.from_numpy(target.astype(np.float32)*scale), \
                torch.from_numpy(model_points.astype(np.float32)*scale), \
                torch.LongTensor([self.entries[index]['class']])
        else:
            return torch.from_numpy(xyz.astype(np.float32)*scale), \
                torch.LongTensor(choose.astype(np.int32)), \
                self.norm(torch.from_numpy(img.astype(np.float32))), \
                torch.from_numpy(target.astype(np.float32)*scale), \
                torch.from_numpy(model_points.astype(np.float32)*scale), \
                torch.LongTensor([self.entries[index]['class']]), \
                transform, self.entries[index]['positions_file']

    def __len__(self):
        return len(self.entries)

    def get_sym_list(self):
        return [0, 1, 2, 3, 4, 5, 6, 7]

    def get_num_points_mesh(self):
        return self.model_points_num


def show_points(pts, title):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', title=title)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker='o')


if __name__ == "__main__":
    dataset = BinDataset(mode='train', dataset_root='Gajdosech_etal_2021_dataset',
                         num_points=1000, width=256, height=256)
    points, choose, img, target, model_points, idx = dataset[457]

    print('points.shape', points.shape)
    print('choose.shape', choose.shape)
    print('img.shape', img.shape)
    print('target.shape', target.shape)
    print('model_points.shape', model_points.shape)
    print('idx', idx)

    save_ply(np.array(target), 'target.ply', text=True)

    show_points(points, 'points')
    show_points(model_points, 'model_points')
    show_points(target, 'target')
    plt.show()
