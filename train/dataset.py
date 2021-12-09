from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from imutils.paths import list_files


class ResizedLMDBDataset(Dataset):
    def __init__(self, path, transform, resolution, is_test, train_ratio):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.train_length = int(self.length * train_ratio)
        self.test_length = int(self.length - self.train_length)

        self.is_test = is_test
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        if not self.is_test:
            return self.train_length
        else:
            return self.test_length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            if not self.is_test:
                key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            else:
                key = f'{self.resolution}-{str(index + self.train_length).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


def set_dataset(type, path, transform, resolution, is_test=False, train_ratio=0.8):
    if type == 'resized_lmdb':
        datatype = ResizedLMDBDataset
    else:
        raise NotImplementedError
    return datatype(path, transform, resolution, is_test, train_ratio)
