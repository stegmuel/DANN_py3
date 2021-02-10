from torch.utils.data import Dataset
import h5py
import abc


class BaseDatasetHDF(Dataset):
    def __init__(self, hdf5_filepath, phase, batch_size, use_cache, cache_size=30):
        """
        Initializes the class BaseDatasetHDF relying on a .hdf5 file that contains the complete data for all phases
        (train, test, validation). It contains the input data as well as the target data to reduce the amount of
        computation done in fly. The samples are first split w.r.t. the phase (train, test, validation) and then w.r.t.
        the status (input, target). A pair of (input, target) samples is accessed with the same index. For a given
        phase a pair is accessed as: (hdf[phase]['input'][index], hdf[phase]['target'][index]).
        The .hdf5 file is stored on disk and only the queried samples are loaded in RAM. To increase retrieval speed
        a small cache in RAM  is implemented. When using the cache, one should note the following observations:
            - The speed will only improve if the data is not shuffled.
            - The cache size must be adapted to the computer used.
            - The number of workers of the data loader must be adapted to the computer used and the cache size.
            - The cache size must be a multiple of the chunk size that was used when filling the .hdf5 file.
        :param hdf5_filepath: location of the .hdf5 file.
        :param phase: phase in which the dataset is used ('train'/'valid'/'test').
        :param batch_size: size of a single batch.
        :param use_cache: boolean indicating if the cache should be used or not.
        :param cache_size: size of the cache in number of batches.
        """
        self.hdf5_filepath = hdf5_filepath
        self.phase = phase
        self.batch_size = batch_size

        # Initialize cache to store in RAM
        self.use_cache = use_cache
        if self.use_cache:
            self.cache = {'input': None, 'target': None}
            self.cache_size = cache_size * batch_size
            self.cache_min_index = None
            self.cache_max_index = None
            self.load_chunk_to_cache(0)

    def __len__(self):
        """
        Returns the total length of the dataset.
        :return: length of the dataset.
        """
        with h5py.File(self.hdf5_filepath, 'r') as hdf:
            length = hdf[self.phase]['input'].shape[0]
        return length

    def is_in_cache(self, index):
        """
        Checks if the queried data is in cache.
        :param index: index of the sample to load.
        :return: boolean indicating if the data is available in cache.
        """
        return index in set(range(self.cache_min_index, self.cache_max_index))

    def load_chunk_to_cache(self, index):
        """
        Loads a chunk of data in cache from disk. The chunk of data is the block of size self.size_cache and contains
        the samples following the current index. This is only efficient if data is not shuffled.
        :param index: index of a single sample that is currently being queried.
        :return: None.
        """
        with h5py.File(self.hdf5_filepath, 'r') as hdf:
            self.cache_min_index = index
            self.cache_max_index = min(len(self), index + self.cache_size)
            self.cache['input'] = hdf[self.phase]['input'][self.cache_min_index: self.cache_max_index]
            self.cache['target'] = hdf[self.phase]['target'][self.cache_min_index: self.cache_max_index]

    @abc.abstractmethod
    def transform(self, x):
        """

        :param x:
        :return:
        """

    @abc.abstractmethod
    def get_image_transformer(self):
        """

        :return:
        """

    @abc.abstractmethod
    def transform(self, x):
        """

        :param x:
        :return:
        """
