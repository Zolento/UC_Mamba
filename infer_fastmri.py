import os
import torch
import argparse
import numpy as np
import sys
sys.path.append(__file__.rsplit('/', 2)[0])
from tqdm import tqdm
from utils.utils import kwargs_vssm_unrolled as kwargs
from models.model_VSSM import VSSMUNet_unrolled
from utils.dataset import FastmriDataset
from utils.loss import compute_metrics2c_full
from torch.utils.data.dataloader import DataLoader

def rebuild(output_list, slice_num):
    p = 0
    out = []
    for i in range(len(slice_num)):
        tmps = []
        for _ in range(slice_num[i]):
            tmps.append(output_list[p])
            p += 1
        out.append(torch.cat(tmps))
    return out

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('-iter', type=int, default=6, required=False, help='iteration')
    parser.add_argument('-d_model', type=int, default=64, required=False, help='d_model')
    parser.add_argument('-dctype', type=str, default='AM', required=False, help='VN;AM')
    parser.add_argument('-eval_model', type=str, default='model_final', required=False, help='eval model name')
    parser.add_argument('-model_folder', type=str, default=None, required=False, help='from which filefolder to infer')
    parser.add_argument('-dataset_name', type=str, default='fastMRI', required=False, help='dataset name')
    parser.add_argument('-data_path', type=str, default='/home/dataset/fastMRI/knee_singlecoil', required=False, help='dataset path')
    parser.add_argument('-pre_name', type=str, default='preprocessed_knee_singlecoil_8x_0.04c320x320', required=False, help='preprossed folder name')
    parser.add_argument('-save_path', type=str, default='/home/UC_Mamba/models/saved', required=False, help='model save path')
    parser.add_argument('-output_path', type=str, default='/home/UC_Mamba/models/output', required=False, help='output path')
    parser.add_argument('--useslice', action='store_true', default=True, required=False, help='[OPTIONAL] if useslice')
    parser.add_argument('--dorefine', action='store_true', default=False, required=False, help='[OPTIONAL] if use refined output')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Fastmri_path = args.data_path
    Model_path = args.save_path
    # Checking paths
    assert args.model_folder is not None, 'A specific model-folder must be given to infer! eg: 2024-01-01-00:00:00preprocessed_knee_singlecoil_8x_0.04c320x320_VSSM_unrolled_AMiterAM3x8'
    Cur_model_path = os.path.join(os.path.join(Model_path, args.dataset_name), args.model_folder)
    if args.dorefine:
        expname = '_refined'
    else:
        expname = ''
    output_path = os.path.join(os.path.join(os.path.join(args.output_path, args.dataset_name), args.model_folder), args.eval_model) + expname
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    infer_dataset = FastmriDataset(rootpath=Fastmri_path, 
                pre_name=args.pre_name,
                name='val',
                useslice=args.useslice,
                infer=True,
                )
    infer_dataloader = DataLoader(dataset=infer_dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=0,
                            pin_memory=True)
    torch.cuda.empty_cache()
    print("Creating model")
    if kwargs['d_model'] != args.d_model:   
        kwargs['d_model'] = args.d_model
        kwargs['UNet_base_num_features'] = args.d_model
    if kwargs['input_channels'] != 2:
        kwargs['input_channels'] = 2
    for k,v in kwargs.items():
        print(k,v)
    model = VSSMUNet_unrolled(iter=args.iter, DC_type=args.dctype, kwargs=kwargs)
    model = model.to(device)
    print("Start evaluation...")
    checkpoint = torch.load(os.path.join(Cur_model_path, args.eval_model+'.pth'), map_location='cuda:0')['model_state_dict']
    name = next(iter(checkpoint))
    if name[:6] == 'module':
        new_state_dict = {}
        for k,v in checkpoint.items():
            new_state_dict[k[7:]] = v
        checkpoint = new_state_dict
    model.load_state_dict(checkpoint)
    model.eval()
    label_full = infer_dataset.fdata_full
    slice_num = infer_dataset.slice_num
    files = infer_dataset.filenames
    output_list = []
    with torch.no_grad():
        with tqdm(total=len(infer_dataloader)) as pbar:
            for batch_idx, (input, label, mask) in enumerate(infer_dataloader):
                torch.cuda.empty_cache()
                out = model(input.to(device), mask.to(device), args.dorefine)
                output_list.append(out.to('cpu'))
                pbar.update(1)
    output_list_full = rebuild(output_list, slice_num)
    if args.useslice:
        label_full = rebuild(label_full, slice_num)
    PSNR_list, SSIM_list = compute_metrics2c_full(output_list_full, label_full)
    result_file = os.path.join(output_path, "results.txt")
    with open(result_file, 'w') as f:
            f.write('This is only a reference.\n')
            for i in range(len(output_list_full)):
                f.write("%s test_psnr: %.4f test_ssim: %.4f \n" % (files[i], PSNR_list[i], SSIM_list[i]))
                print("%s test_psnr: %.4f test_ssim: %.4f " % (files[i], PSNR_list[i], SSIM_list[i]))
            f.write("mean_test_psnr: %.4f mean_test_ssim: %.4f \n" % (sum(PSNR_list)/len(PSNR_list), sum(SSIM_list)/len(SSIM_list)))
    print("mean_test_psnr: %.4f mean_test_ssim: %.4f " % (sum(PSNR_list)/len(PSNR_list), sum(SSIM_list)/len(SSIM_list)))
    print("Saving...")
    for i in range(len(output_list_full)):
            np.save(output_path+'/'+files[i][:-3], output_list_full[i])
            # np.save(output_path+'/'+files[i]+'_fi', label_full[i])
    print("Inference done.")