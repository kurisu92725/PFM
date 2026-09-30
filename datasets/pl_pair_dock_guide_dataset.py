from .pl_pair_dataset import PocketLigandPairDataset


class PocketLigandPairDockGuideDataset(PocketLigandPairDataset):
    def __init__(self, raw_path, transform=None, version='final', index_path=None):
        super().__init__(raw_path, transform=transform, version=f'dock_guide_{version}')
