import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

from utils import read_PolyData, PolyDataToNumpy


class Dataset(InMemoryDataset):
    '''
    Dataset class. See more at: 
    https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html

    Args:

    root (string, optional): Root directory where the dataset should be
            saved. (optional: :obj:`None`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)

    data (list, optional): A list of data objects to be represented as 
        :obj:`torch_geometric.data.Data`. The list elements should have
        a following structure:
        {
            'path': (string, required): path to the vtk object.
            'y: (int, required): class label.
            ... all other attributes will be passed to  
            :obj:`torch_geometric.data.Data` instance.
        }
        (default: :obj:`None`)
    '''

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        data=None
    ):
        self.data = data
        super(Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [x['path'] for x in self.data]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        print('Processing data ...')

        data_list = []
        for data_obj in tqdm(self.data):
            data_path = data_obj['path']
            label = data_obj['y']
            poly = read_PolyData(data_path)
            verts, tris = PolyDataToNumpy(poly)
            extra_kwargs = get_extra_kwargs(data)
            data = Data(
                pos=torch.tensor(verts),
                face=torch.tensor(tris.T),
                y=torch.tensor([label]),
                **extra_kwargs
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def get_extra_kwargs(data):
    '''
    Function to return a dict arguments without 'path' and 'y' fields.

    Args:
        data: (dict, required): A dict representation of the data object
    '''

    data_copy = data.copy()
    del data_copy['path']
    del data_copy['y']
    return data_copy
