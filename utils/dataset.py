import torch
import numpy as np
import fastmri
import os
from natsort import natsorted
from torch.utils.data.dataset import Dataset
from pathlib import Path
from tqdm import tqdm

class FastmriDataset(Dataset):
    def __init__(self, rootpath, pre_name, name, infer=False, dram=True, useslice=True):
        super(FastmriDataset, self).__init__()
        self.dataset = []
        self.fdata_full = []
        if name == 'val':
            datainfo = np.load(os.path.join(rootpath, (name + "/datainfo.npy")))
            self.filenames = datainfo[0]
            self.slice_num = [int(i) for i in datainfo[1]]
        self.dram = dram
        pre_name = name + '/' + pre_name
        if dram==False and useslice==False:
            useslice = True
            print("Sliced data reading supported only, force useslice=True when not dram")
        # now get filepath
        # for self.dram, get filepath in /fi/
        # for not self.dram, get filepath in /fislice/
        if self.dram:
            if useslice:
                path = Path(os.path.join(rootpath, (pre_name+"/fislice/")))
            else:
                path = Path(os.path.join(rootpath, (pre_name+"/fi/")))
        else:
            path = Path(os.path.join(rootpath, (pre_name+"/fislice/")))
        files = natsorted([file.name for file in path.rglob("*.npy")])
        files = tqdm(files)
        print("Start reading fastmri, toltal %s number: %s" % (name, str(len(files))))
        for f in files:
            if useslice:
                ukpath = os.path.join(rootpath, pre_name)+"/ukslice/" + f.replace('fi_', 'uk_')
                fipath = os.path.join(rootpath, pre_name)+"/fislice/" + f
                maskpath = os.path.join(rootpath, pre_name)+"/maskslice/" + f.replace('fi_', 'mask_')
            else:
                ukpath = os.path.join(rootpath, pre_name)+"/uk/" + f.replace('_fi', '_uk')
                fipath = os.path.join(rootpath, pre_name)+"/fi/" + f
                maskpath = os.path.join(rootpath, pre_name)+"/mask/" + f.replace('_fi', '_mask')
            if self.dram:
                fi = torch.tensor(np.load(fipath))#(kz,kx,ky,2);
                mask = torch.tensor(np.load(maskpath))#(1,1,kx,ky);
                uk = torch.tensor(np.load(ukpath))#(1/kz,kx,ky,2);
            else:
                ukpath = os.path.join(rootpath, pre_name)+"/ukslice/" + f.replace('fi_', 'uk_')
                fipath = os.path.join(rootpath, pre_name)+"/fislice/" + f
                maskpath = os.path.join(rootpath, pre_name)+"/maskslice/" + f.replace('fi_', 'mask_')
            if infer:
                fi = torch.tensor(np.load(fipath))
                if useslice:
                    self.fdata_full.append(fi.unsqueeze(0))#(kx,ky)
                else:
                    self.fdata_full.append(fastmri.complex_abs(fi))#(z,kx,ky)
            if self.dram:
                if useslice:
                    self.dataset.append([uk, fi, mask])
                else:
                    for i in range(uk.shape[0]):
                        uki = uk[i,...].unsqueeze(0)
                        fii = fastmri.complex_abs(fi[i,...])
                        self.dataset.append([uki, fii, mask])
            else:
                self.dataset.append([ukpath, fipath, maskpath])
            files.set_description("Reading processed data/datapath %s" % f)
            
    def __getitem__(self, index):
        if self.dram:
            uki, fii, mask = self.dataset[index]
        else:
            ukpath, fipath, maskpath = self.dataset[index]
            uki = torch.tensor(np.load(ukpath))
            fii = torch.tensor(np.load(fipath))
            mask = torch.tensor(np.load(maskpath))
        return uki, fii, mask
    
    def __len__(self):
        
        return len(self.dataset)
    