from loader.carbon_dataset import CarbonSpectraDataset

if __name__ == '__main__':
    dataset = CarbonSpectraDataset(root='D:\\project\\Nmr-Spectra-Predict-Model\\data')
    
    print(len(dataset))