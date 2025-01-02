# -*- coding: utf-8 -*-
import re
import os
import torch

from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset

from utils.dataset import (
    ExtractFluorineShift, 
    MolToGraph, 
    MolToFingerprints, 
    GenerateSequentialSmiles,
    GernerateMask
)


# Disable rdkit warnings
RDLogger.DisableLog('rdApp.*')


class FluorineSpectraDataset(InMemoryDataset):
    def __init__(
        self, 
        root = None, 
        transform = None, 
        pre_transform = None, 
        pre_filter = None, 
    ):
        self.root = root
    
        super(FluorineSpectraDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'nmrshiftdb2withsignals.sd'

    @property
    def processed_file_names(self):
        return 'nmr_19F_data_processed.pt'

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root,'processed')
    
    def process(self):
        # 读取原始数据
        suppl = Chem.SDMolSupplier(
            os.path.join(self.raw_dir, self.raw_file_names),
            removeHs = False,
            sanitize = True
        )
        
        fluorine_mols = []
        pattern = re.compile(r'Spectrum 19F(\s[0-9])?')
        # 如果分子中包含氟原子，则加入到列表中
        for mol in tqdm(suppl, desc="Searching 19F data", total=len(suppl)):
            if mol is not None and mol.GetNumAtoms() > 0:
                # 如果所有的 keys 中 有 'Spectrum 19F' 字符串，则加入到列表中
                if any(pattern.search(key) for key in mol.GetPropsAsDict().keys()):
                    fluorine_mols.append(mol)
                    print(f"Find a molecule with fluorine atoms.")
            else:
                continue
     
        print(f"Total {len(fluorine_mols)} molecules with fluorine atoms.")

        data_list = []
        
        # 读取数据
        for i, mol in tqdm(enumerate(fluorine_mols), desc="Processing data", total=len(fluorine_mols)):
            # 提取氟谱数据
            fluorine_shift = ExtractFluorineShift(mol)
            
            mask, shift = GernerateMask(mol, fluorine_shift)
            
            data = Data()
            
            # 提取分子图
            graph = MolToGraph(mol)
            finger_print = MolToFingerprints(mol)
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            smiles_vector = GenerateSequentialSmiles(smiles)
            
            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.tensor(shift, dtype=torch.float32)
            data.smiles_vector = torch.tensor(smiles_vector, dtype=torch.int64)
            data.fingerprint = torch.tensor(finger_print, dtype=torch.int64)
            data.mask = torch.tensor(mask, dtype=torch.bool)
            data.smiles = smiles

            data_list.append(data)
            
        torch.save(self.collate(data_list), self.processed_paths[0])


