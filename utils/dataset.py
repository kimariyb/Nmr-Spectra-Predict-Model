import numpy as np

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from pubchemfp import GetPubChemFPs


# Disable rdkit warnings
RDLogger.DisableLog('rdApp.*')


# 原子特征允许的集合
ALLOWED_ATOMS = {
    'atomic_num_list': list(range(1, 119)) + ['other'],
    'chirality_list': [        
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'other'
    ],
    'degree_list': list(range(0, 11)) + ['other'],
    'charge_list': list(range(-5, 6)) + ['other'],
    'num_H_list': list(range(0, 9)) + ['other'],
    'num_radical_e_list': list(range(0, 5)) + ['other'],
    'valence_list': list(range(0, 9)) + ['other'],
    'hybridization_list': [
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'other',
    ],
    'is_aromatic_list': [False, True],
    'is_in_ring_list': [False, True],
}

# 键特征允许的集合
ALLOWED_BONDS = {
    'type_list': [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated_list': [False, True],
    'is_in_ring_list': [False, True],
}


def ExtractCarbonShift(mol):
    r"""
    提取 13C 化学位移信息
    """
    # 获取分子的属性字典
    mol_props = mol.GetPropsAsDict()
    atom_shifts = {}
    
    # 遍历所有属性键
    for key in mol_props.keys():
        # 找到以 'Spectrum 13C' 开头的属性
        if key.startswith('Spectrum 13C'):
            # 分割属性值，获取每个化学位移信息
            for shift in mol_props[key].split('|')[:-1]:
                # 分割化学位移值、未知字段和原子索引
                [shift_val, _, shift_idx] = shift.split(';')
                shift_val, shift_idx = float(shift_val), int(shift_idx)
                # 如果原子索引不在字典中，初始化为空列表
                if shift_idx not in atom_shifts: 
                    atom_shifts[shift_idx] = []
                    
                # 将化学位移值添加到对应原子索引的列表中
                atom_shifts[shift_idx].append(shift_val)

    # 对每个原子索引，计算化学位移值的中位数
    for j in range(mol.GetNumAtoms()):
        if j in atom_shifts:
            atom_shifts[j] = np.median(atom_shifts[j])

    return atom_shifts


def ExtractHydrogenShift(mol):
    r"""
    提取 1H 化学位移信息
    """
    # 获取分子的属性字典
    mol_props = mol.GetPropsAsDict()
    atom_shifts = {}
    
    # 遍历所有属性键
    for key in mol_props.keys():
        # 找到以 'Spectrum 1H' 开头的属性
        if key.startswith('Spectrum 1H'):
            tmp_dict = {}
            # 分割属性值，获取每个化学位移信息
            for shift in mol_props[key].split('|')[:-1]:
                # 分割化学位移值、未知字段和原子索引
                [shift_val, _, shift_idx] = shift.split(';')
                shift_val, shift_idx = float(shift_val), int(shift_idx)
                # 如果原子索引不在字典中，初始化为空列表
                if shift_idx not in atom_shifts: 
                    atom_shifts[shift_idx] = []
                # 如果原子索引不在临时字典中，初始化为空列表
                if shift_idx not in tmp_dict: 
                    tmp_dict[shift_idx] = []
                tmp_dict[shift_idx].append(shift_val)
                
            # 将临时字典中的值添加到原子位移字典中
            for shift_idx in tmp_dict.keys():
                atom_shifts[shift_idx].append(tmp_dict[shift_idx])
                
    # 对每个原子索引，处理化学位移值
    for shift_idx in atom_shifts.keys():
        # 找到每个原子索引的最大位移列表长度
        max_len = np.max([len(shifts) for shifts in atom_shifts[shift_idx]])
        
        for i in range(len(atom_shifts[shift_idx])):
            # 如果位移列表长度小于最大长度，进行填充
            if len(atom_shifts[shift_idx][i]) < max_len:
                # 如果列表长度为1，重复该值填充到最大长度
                if len(atom_shifts[shift_idx][i]) == 1:
                    atom_shifts[shift_idx][i] = [atom_shifts[shift_idx][i][0] for _ in range(max_len)]
                # 如果列表长度大于1，使用均值填充到最大长度
                elif len(atom_shifts[shift_idx][i]) > 1:
                    while len(atom_shifts[shift_idx][i]) < max_len:
                        atom_shifts[shift_idx][i].append(np.mean(atom_shifts[shift_idx][i]))

            # 对位移列表进行排序
            atom_shifts[shift_idx][i] = sorted(atom_shifts[shift_idx][i])
        # 计算每个原子索引的位移值的中位数
        atom_shifts[shift_idx] = np.median(atom_shifts[shift_idx], 0).tolist()
    
    return atom_shifts


def ExtractMolData(mol):
    r"""
    提取 SMILES 字符串和分子编号
    """
    # 将 RDKit 分子对象转换为 SMILES 字符串
    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    # 同时得到对应的分子编号
    mol_props = mol.GetPropsAsDict()
    mol_id = mol_props['nmrshiftdb2 ID']

    return {'smiles': smi, 'mol_id': mol_id}


def SafeIndex(allowed_list, element):
    r"""
    Return index of element e in list l. 
    If e is not present, return the last index
    
    Parameters
    ----------  
    allowed_list: list
        允许的元素列表
    element: any
        待查找的元素
    """
    try:
        return allowed_list.index(element)
    except:
        return len(allowed_list) - 1


def AtomToFeature(atom):
    r"""
    将 RDKIT 原子对象转换为特征向量
    """
    atom_feature = []
    
    atom_feature = [
        SafeIndex(ALLOWED_ATOMS['atomic_num_list'], atom.GetAtomicNum()),
        SafeIndex(ALLOWED_ATOMS['chirality_list'], str(atom.GetChiralTag())),
        SafeIndex(ALLOWED_ATOMS['degree_list'], atom.GetTotalDegree()),
        SafeIndex(ALLOWED_ATOMS['charge_list'], atom.GetFormalCharge()),
        SafeIndex(ALLOWED_ATOMS['num_H_list'], atom.GetTotalNumHs()),
        SafeIndex(ALLOWED_ATOMS['num_radical_e_list'], atom.GetNumRadicalElectrons()),
        SafeIndex(ALLOWED_ATOMS['valence_list'], atom.GetTotalValence()),
        SafeIndex(ALLOWED_ATOMS['hybridization_list'], str(atom.GetHybridization())),
        ALLOWED_ATOMS['is_aromatic_list'].index(atom.GetIsAromatic()),
        ALLOWED_ATOMS['is_in_ring_list'].index(atom.IsInRing()),
    ]
    
    return atom_feature


def BondToFeature(bond): 
    r"""
    将 RDKIT 键对象转换为特征向量
    """
    bond_feature = []
    
    bond_feature = [
        SafeIndex(ALLOWED_BONDS['type_list'], str(bond.GetBondType())),
        ALLOWED_BONDS['stereo_list'].index(str(bond.GetStereo())),
        ALLOWED_BONDS['is_conjugated_list'].index(bond.GetIsConjugated()),
        ALLOWED_BONDS['is_in_ring_list'].index(bond.IsInRing()),
    ]
    
    return bond_feature


def MolToGraph(mol):
    r"""
    将分子转换为图数据
    """
    # 获取分子的原子特征
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(AtomToFeature(atom))

    x = np.array(atom_features, dtype=np.int64)
    z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    
    # 获取分子的边特征
    num_bond_features = 4 # bond type, is conjugated, is rings, bond stereo
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edge_feature =  BondToFeature(bond)
            
            edges_list.append((i, j))
            edge_features.append(edge_feature)
            edges_list.append((j, i))
            edge_features.append(edge_feature)
        
        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features, dtype=np.int64)
        
    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)
        
    # 构建图数据         
    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)
    graph["z"] = z
    
    return graph


def GenerateSequentialSmiles(smile):
    r"""
    序列化 SMILES 字符串
    """
    smi_to_seq = "(.02468@BDFHLNPRTVZ/bdfhlnprt#*%)+-/13579=ACEGIKMOSUWY[]acegimosuy\\"
    seq_dict_smi = {v: (i + 1) for i, v in enumerate(smi_to_seq)}  ## 对照字典，通过循环得到每个字符对应的index
    max_seq_smi_len = 100
    
    x = np.zeros(shape=max_seq_smi_len)
    for i, ch in enumerate(smile[:max_seq_smi_len]):
        x[i] = seq_dict_smi[ch]
        
    return np.array(x)


def MolToFingerprints(mol):
    r"""
    将分子转换为指纹
    """
    fp = []

    fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)  ## maccs指纹
    fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)  ## ErGFingerprint指纹
    fp_pubcfp = GetPubChemFPs(mol)  
    fp.extend(fp_maccs)
    fp.extend(fp_phaErGfp)
    fp.extend(fp_pubcfp)

    return np.array(fp)


if __name__ == '__main__':
    # 读取分子数据
    mol = Chem.MolFromSmiles('CCO')
    # 将分子转换为图数据
    graph = MolToGraph(mol)
    fingerprint = MolToFingerprints(mol)
    print(graph)
    print(len(fingerprint))
    
    # 绘制 RDKit 分子对象
    from rdkit.Chem import Draw
    img = Draw.MolToImage(mol)
    img.show()