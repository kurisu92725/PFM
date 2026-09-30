import os
import pickle

import lmdb
from torch.utils.data import Dataset

from .pl_data import ProteinLigandData


class PocketLigandPairDataset(Dataset):
    def __init__(self, raw_path, transform=None, version='final'):
        super().__init__()
        if raw_path.endswith('.lmdb'):
            self.processed_path = raw_path
        else:
            self.processed_path = os.path.join(
                os.path.dirname(raw_path),
                os.path.basename(raw_path) + f'_processed_{version}.lmdb',
            )
        if not os.path.isfile(self.processed_path):
            raise FileNotFoundError(
                f'Preprocessed dataset not found: {self.processed_path}'
            )
        self.transform = transform
        self.db = None
        self.keys = None

    def _connect_db(self):
        self.db = lmdb.open(
            self.processed_path,
            subdir=False,
            create=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        data = pickle.loads(self.db.begin().get(self.keys[idx]))
        data = ProteinLigandData(**data)
        data.id = idx
        if data.protein_pos.size(0) == 0:
            raise ValueError(f'Pocket {idx} has no protein atoms.')
        if self.transform is not None:
            data = self.transform(data)
        return data
