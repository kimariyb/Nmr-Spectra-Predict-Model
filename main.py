from loader.fluorine_dataset import FluorineSpectraDataset
from rdkit import Chem
from rdkit.Chem import Draw


if __name__ == '__main__':
    dataset = FluorineSpectraDataset(root='D:\\project\\Nmr-Spectra-Predict-Model\\data')
    data = dataset[1]
    print(data)
    print(data.mask)
    print(data.y)
    
    mol = Chem.MolFromSmiles(data.smiles)
    img =  Draw.MolToImage(mol)
    img.show()