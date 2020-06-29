# vtk-to-pytorch-geometric

Generate point cloud representation of the `.vtk` 3d object. This representation is compatible with PyTorch Geometric library.

## How to

### Read your data

First implement `read_data_objects` function in `utils.py`. You need to represent each 3d object as a dict with following attributes:

```python
{
  'path': (string): absolute path to this .vtk object
  'y': (int): label of the object
  ... and optional attributes
}
```
Note that you can pass more attributes to the dict. Those will be automatically passed to the PyTorch Geometric abstraction as well. 


### Generate and save your PyTorch Geometric Dataset

To generate the dataset use `generate.py` script

```bash
python generate.py \
--path /path/to/dataset \
--num_points 1024
```
Args:

- `path` arguments means where your dataset will be stored. Dataset generated to `path` just once. All instances of Dataset object (e.g. in your train code) with existing path will just read the data instead of processing it.
- `num_points` Number of points uniformly sampled from the shape surface
