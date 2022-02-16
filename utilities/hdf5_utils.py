
import h5py
from numpy import ndarray, isscalar, array

def read_hdf5_into_dict(group, read_datasets='all', read_groups='all', read_paths='all', read_attrs=True):
    '''
    ----------------------------------------------------------------------------
    Given an open HDF5 group (i.e. from open `h5py.File`), reads the data
    stucture recursively and stores in nested Python `dict` objects.
    
    `group` -- HDF5 group object, e.g. the root HDF5 file `h5py.File`
    `read_datasets` -- given a list of datasets to read, if a `Dataset.name`
        matches an entry in `read_datasets`, then it will be read.  Otherwise,
        if `'all'` (default) this function will parse all datasets
    `read_groups` -- given a list of group names to read, if a `Group.name`
        matches an entry in `read_groups`, then that group will be read.
        Otherwise, if `'all'` (default) this function will parse all datasets
    `read_paths` -- given a list of paths to either groups or datasets, this
        function will read the data starting from these paths.  Otherwise, if
        `'all'`, then all paths will be read.
    `read_attrs` -- whether will read attributes.  This function only reads and
        sets values for attributes of groups, since there is not clear key using
        this schema for attributes of datasets.
    ''' 
    data = {}
    # read group attributes
    if read_attrs:
        for key, attr in group.attrs.items():
            data[key] = attr
    if read_paths == 'all':
        for key in group.keys():
            if isinstance(group[key], h5py.Dataset) and (read_datasets == 'all' or key in read_datasets):
                if group[key].shape == ():
                    data[key] = group[key][()]
                else:
                    data[key] = group[key][:]
            elif isinstance(group[key], h5py.Group) and (read_groups == 'all' or key in read_groups):
                data[key] = read_hdf5_into_dict(group[key], read_datasets, read_groups)
    else:
        for path in read_paths:
            if path not in group:
                continue  # path must be valid within group
            # Note: valid HDF5 paths do NOT begin with '/'
            keys = path.split('/')  # there must be at least one valid string in the path
            assert(len(keys) >= 1)
            if len(keys) == 1: 
                if isinstance(group[path], h5py.Dataset):
                    if group[path].shape == ():
                        data[keys[0]] = group[path][()]
                    else:
                        data[keys[0]] = group[path][:]
                elif isinstance(group[path], h5py.Group):
                    data[keys[0]] = read_hdf5_into_dict(group[path], read_datasets, read_groups)
            else:
                data[keys[0]] = d = {}
                for k in keys[1:-1]:
                    d[k] = {}
                    d = d[k]
                if isinstance(group[path], h5py.Dataset):
                    if group[path].shape == ():
                        data[keys[-1]] = group[path][()]
                    else:
                        data[keys[-1]] = group[path][:]
                elif isinstance(group[path], h5py.Group):
                    d[keys[-1]] = read_hdf5_into_dict(group[path], read_datasets, read_groups)
    return data


def write_dict_to_hdf5(data, f, path='/'):
    '''
    ----------------------------------------------------------------------------
    Given a dictionary-like structure, writes to an HDF5 file.  `ndarray`
    objects are written as HDF5 datasets, while scalars, strings, etc. are
    written as group attributes.  The HDF5 path structure follows the tree
    structure of the dictionary.
    
    `data` -- dictionary object containing data
    `f` -- handle to HDF5 output file
    '''
    for key, item in data.items():
        if isscalar(item):
            f.create_dataset(path + str(key), data=item)
        elif isinstance(item, ndarray) or isinstance(item, list) or isinstance(item, tuple):
            f.create_dataset(path + str(key), data=array(item))
        elif isinstance(item, dict):
            write_dict_to_hdf5(item, f, path + str(key) + '/')
        else:
            raise ValueError('Cannot save {0} type'.format(type(item)))
