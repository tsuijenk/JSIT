from scipy.io import savemat
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tifffile import imread
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from argparse import ArgumentParser
from typing import Optional


'''
    Change format from (round,x,y) to (x,y,round)
'''
def aligned_jsit(img:str,out:str):
    i = imread(img)
    oi = np.stack(i,axis=-1)
    savemat(f'{out}.mat',{'ans':oi})
    return f"Wrote {os.path.split(img)[1]}"
'''
    Helper to convert C1E1 barcode strings to codebook matrix
'''
def split_barcode(barcode:str):
    return [int(bit) for bit in barcode.split()]


def main(img_folder:str,output_folder:str,*, workers:int = 7, codebook: Optional[str] = None):
    if not os.path.exists(output_folder): os.mkdir(output_folder)
    #save Code Book Skip first for rows bc of codebook weirdness
    
    if codebook is not None:
        c = pd.read_csv(
            codebook,skiprows=4,header = None
        )
        barcodes = c.iloc[:,2].apply(split_barcode)
        barcodes = np.array([np.array(barcode) for barcode in barcodes])
        savemat(
            os.path.join(output_folder,'codebook.mat'),
            {'ans':barcodes}
        )

    print("Converting FOV's to JSIT format")
    
    imgs = [f for f in os.listdir(img_folder) 
            if os.path.isfile(os.path.join(img_folder, f)) and f.lower().endswith(('.tif', '.tiff'))]
    
    with ThreadPoolExecutor(max_workers=workers) as exec:

        jobs  = []

        for fov in imgs:
            img = os.path.join(img_folder,fov)
            out = os.path.join(output_folder,fov)
            jobs.append(exec.submit(aligned_jsit,img,out))
            
        for job in jobs:
            print(job.result())
        
        
if __name__ == "__main__":
    cli = True
    if cli: 
        parser = ArgumentParser()
        parser.add_argument('-i',type=str, help="Folder for Restacked Aligned images")
        parser.add_argument('-c', type=str, default=None, help='path to codebook.  Expects similar format to C1E1_codebook.csv')
        parser.add_argument('-o', default="JSIT_input",type=str, help="Folder to write images and codebook to")
        args = parser.parse_args()
        main(img_folder=args.i,codebook=args.c,output_folder=args.o)
    else:
        im ="/home/isaac/dev/molonc/gsc/scratch/restacked_aligned"
        cb = "/home/isaac/dev/molonc/gsc/scratch/parameters/codebooks/C1E1_codebook.csv"
        main(im,cb,'xp6873')


    """
    python aligned_stack_jsit.py -i "/projects/molonc/scratch/jtsui/XP8054/serval_restacked_aligned_preprocessed_images" -o "/projects/molonc/scratch/jtsui/JSIT_experiment/reformatted_for_JSIT/XP8054/restacked_aligned_preprocessed"
    
    python aligned_stack_jsit.py -i "/projects/molonc/roth_lab/jtsui/XP6873/serval_restacked_aligned_preprocessed_images" -o "/projects/molonc/scratch/jtsui/JSIT_experiment/reformatted_for_JSIT/XP6873/restacked_aligned_preprocessed"

    """
