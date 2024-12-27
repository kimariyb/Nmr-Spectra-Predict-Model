# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.BRICS import FindBRICSBonds
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

from utils.dataset import ExtractCarbonShift, MolToGraph

# Disable rdkit warnings
RDLogger.DisableLog('rdApp.*')


class CarbonSpectraDataset(InMemoryDataset):
    def __init__(
        self, 
        root = None, 
        transform = None, 
        pre_transform = None, 
        pre_filter = None, 
    ):
        self.root = root
    
        super(CarbonSpectraDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'nmrshiftdb2withsignals.sd'

    @property
    def processed_file_names(self):
        return 'nmr_13C_data_processed.pt'

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root,'processed')
    
    def process(self):
        ...
        
    def mol2graph(self, mol):
        ...