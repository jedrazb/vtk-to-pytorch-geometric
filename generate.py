import argparse

from dataset import Dataset
from utils import read_data_objects

import torch_geometric.transforms as T

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--num_points', required=False, type=int, default=1024)
    dataset_args = parser.parse_args()

    num_points = dataset_args.num_points

    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(num_points)

    data_objects = read_data_objects()

    Dataset(
        dataset_args.path,
        transform,
        pre_transform,
        data_objects
    )
