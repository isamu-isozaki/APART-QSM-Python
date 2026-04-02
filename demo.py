import os, sys, time
from math import ceil

import nibabel as nib
import SimpleITK as sitk
import pyqtgraph as pg
import numpy as np
import scipy.io as scio

from functions.crosshairView import crosshairView
from functions.parser import parse_int, parse_float, parse_array, array_to_str
from functions.apart_qsm import *
import time
import torch
from STISuite_pytorch.qsm_funcs import Load_QSM
from STISuite_pytorch.qsm_funcs import create_mask
from STISuite_pytorch.qsm_funcs import MR_phase_unwrap
from STISuite_pytorch.qsm_funcs import v_sharp_echoes
from STISuite_pytorch.qsm_funcs import qsm_star

def Load(filePath):
    cur_series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(filePath)
    qsm_idx = []
    for idx in cur_series_ids:
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(filePath, idx)
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        series_reader.SetMetaDataDictionaryArrayUpdate(True)
        image3D = series_reader.Execute()
        image3D = sitk.DICOMOrient(image3D, 'LPS')
        assert image3D.GetDimension() == 3
        description = series_reader.GetMetaData(0, '0008|103e').strip()
        if 'qsm' in description.lower() or 'eswan' in description.lower() or 'swi' in description.lower():
            qsm_idx.append(idx)
            continue
    if len(qsm_idx) != 0:
        raw_data, voxel_size, matrix_size, CF, delta_TE, TE, affine_3D, B0_dir, B0, origin = Load_QSM(
                filePath, qsm_idx)
    else:
        raise ValueError('No data found in the provided DICOM directory.')
        
    return raw_data, voxel_size, matrix_size, CF, delta_TE, TE, affine_3D, B0_dir, B0, origin
    


if __name__ == '__main__':
    start0 = time.time()

    # ####gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    input_path = "./data"
    dcm_path = os.path.join(input_path, 'GRE')
    t2_path = os.path.join(input_path, 'T2.nii.gz')
    output_path = "./Results"
    os.makedirs(output_path, exists_ok=True)
    t2 = nib.load(t2_path).get_fdata()
    #raw_data, voxel_size, matrix_size, CF, delta_TE, TE, affine_3D, B0_dir, B0, origin = Load(dcm_path)
    raw_data, voxel_size, matrix_size, CF, delta_TE, TE, affine_3D, B0_dir, B0, origin = Load_QSM(dcm_path)
    magnitude = np.abs(raw_data).detach().cpu().numpy()
    phase = np.angle(raw_data)
    affine = np.eye(4)
    affine[:3, :3] = np.diag(np.array(voxel_size, dtype=np.float32))     
    smv_size = 12
    qsm_star_tau = 1e-6
    mask = create_mask(raw_data)
    unwrapped_phase = MR_phase_unwrap(torch.angle(raw_data.to(device)), voxel_size)
    tissue_phase, new_mask = v_sharp_echoes(unwrapped_phase, mask, voxel_size)

    if tissue_phase.shape[3] > 6:
        qsm = qsm_star(tissue_phase[:,:,:,2:].mean(3), new_mask[:,:,:,2],
                                    voxel_size, 
                                    B0_dir, 
                                    B0, 
                                    TE[2:].mean())
        qsm=-qsm
        TE_phi = TE[2:].detach().cpu().numpy().copy()
        phase_shape = tissue_phase[...,2:].detach().cpu().numpy().shape
    else:
        qsm = qsm_star(tissue_phase[:,:,:,2:].mean(3), new_mask[:,:,:,0], 
                                    voxel_size, 
                                    B0_dir, 
                                    B0, 
                                    TE.mean(), 
                    ) 
        qsm = -qsm.detach().cpu().numpy()
        TE_phi = TE.detach().cpu().numpy().copy()
        phase_shape = tissue_phase.detach().cpu().numpy().shape    
    nib.save(nib.Nifti1Image(qsm,affine),os.path.join(output_path, 'qsm.nii.gz'))
    nib.save(nib.Nifti1Image(magnitude,affine),os.path.join(output_path, 'mag.nii.gz'))
    end0 = time.time()
    #### APART-QSM
    start = time.time()
    params_input = build_params_input(
    TE_mag=TE.detach().cpu().numpy(),
    TE_phi=TE_phi,
    voxel_size= np.array(voxel_size, dtype=np.float32),
    B0=B0,
    B0_dir=np.array(B0_dir, dtype=np.float32),
    newmask=new_mask.detach().cpu().numpy(),
    phase_shape=phase_shape,
    mag_shape = magnitude.shape,
    qsm=qsm,
    magnitude=magnitude,
    voxel_size_used=voxel_size,
    t2_map=t2,
    matrix_size=magnitude.shape[:3])

    recon = ARART_Recon(
    mag=magnitude,
    phi_tissue=tissue_phase.detach().cpu().numpy(),
    QSM=qsm,
    params=params_input,
    t2_map=t2,
    output_path=output_path
    )
    Res_map = recon.run()
    end = time.time()
    print("QSM running time: {:.2f} s".format(end0 - start0))
    print("apartQSM running time: {:.2f} s".format(end - start))
    print("total running time: {:.2f} s".format(end - start0))

