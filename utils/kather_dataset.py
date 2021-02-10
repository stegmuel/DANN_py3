from torchvision.transforms import RandomApply, Compose, ColorJitter, RandomGrayscale, RandomRotation, \
    RandomResizedCrop, ToTensor
from utils.base_dataset import BaseDatasetHDF
from PIL.Image import fromarray
import torch
import h5py


class KatherHDF(BaseDatasetHDF):
    def __init__(self, hdf5_filepath, phase, batch_size, use_cache, cache_size=30):
        """
        Initializes the class Kather19HDF relying on a .hdf5 file that contains the complete data for all phases
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
        super(KatherHDF, self).__init__(hdf5_filepath, phase, batch_size, use_cache, cache_size)
        # Get the image transformer
        self.transformer = self.get_image_transformer()

    def get_image_transformer(self):
        transformations = [
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.1),
            RandomGrayscale(p=0.5),
            RandomRotation(degrees=[0., 45.]),
            RandomResizedCrop(size=[224, 224], scale=(0.3, 1.0))]
        return Compose([RandomApply(transformations, p=0.7), ToTensor()])

    def transform(self, x):
        if self.phase == 'train':
            x = self.transformer(fromarray(x))
        else:
            x = ToTensor()(x)
        return x

    def __getitem__(self, index):
        """
        Loads a single pair (input, target) data.
        :param index: index of the sample to load.
        :return: queried pair of (input, target) data.
        """
        if self.use_cache:
            if not self.is_in_cache(index):
                self.load_chunk_to_cache(index)
            x_input = self.cache['input'][index - self.cache_min_index]
            x_target = self.cache['target'][index - self.cache_min_index][None]
        else:
            with h5py.File(self.hdf5_filepath, 'r') as hdf:
                x_input = hdf[self.phase]['input'][index]
                x_target = hdf[self.phase]['target'][index][None]
        return self.transform(x_input), torch.from_numpy(x_target)
