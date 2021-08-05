'''
Predict binding sites given a .pdb file of a protein
'''
from Bio.PDB import PDBParser, PDBIO, Select
import Bio
import os
import sys
import re
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import imp
import molgrid
import argparse
import time
from skimage.morphology import binary_dilation
from skimage.morphology import cube
from skimage.morphology import closing
from skimage.segmentation import clear_border
from skimage.measure import label
import struct
from clean_pdb import clean_pdb
from get_centers import get_centers
from types_and_gninatyper import gninatype,create_types
from model import Model
from rank_pockets import test_model
from unet import Unet
from segment_pockets import test
import gc
def parse_args(argv=None):
    '''Return argument namespace and commandline'''
    parser = argparse.ArgumentParser(description='predict ligand binding site from .pdb file')
    parser.add_argument('-c', '--class_checkpoint', type=str, required=True,
                        help="classification checkpoint")
    parser.add_argument('-s', '--seg_checkpoint', type=str, required=True,
                        help="segmentation checkpoint")
    parser.add_argument('-p','--protein', type=str, required=False, help="pdb file for predicting binding sites")
    parser.add_argument('-r', '--rank', type=int, required=False,
                        help="number of pockets to segment", default=1)
    parser.add_argument('--upsample', type=str, required=False,
                        help="Type of Upsampling", default=None)
    parser.add_argument('--num_classes', type=int, required=False,
                        help="Output channels for predicted masks, default 1", default=1)
    parser.add_argument('-t', '--threshold', type=float, required=False,
                        help="threshold for segmentation", default=0.5)
    parser.add_argument('--mask_dist', type=float, required=False,
                        help="distance from mask to residues", default=3.5)
    args = parser.parse_args(argv)

    argdict = vars(args)
    line = ''
    for (name, val) in list(argdict.items()):
        if val != parser.get_default(name):
            line += ' --%s=%s' % (name, val)

    return (args, line)

def get_model_gmaker_eprovider(test_types,batch_size,model,checkpoint,dims=None):
    model.cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    eptest_large = molgrid.ExampleProvider(shuffle=False, stratify_receptor=False, labelpos=0,balanced=False,iteration_scheme=molgrid.IterationScheme.LargeEpoch,default_batch_size=batch_size)
    eptest_large.populate(test_types)
    if dims is None:
        gmaker = molgrid.GridMaker()
    else:
        gmaker = molgrid.GridMaker(dimension=dims)
    return model, gmaker,  eptest_large

if __name__ == '__main__':
    (args, cmdline) = parse_args()
    #clean pdb file and remove hetero atoms/non standard residues
    protein_file=args.protein
    protein_nowat_file=protein_file.replace('.pdb','_nowat.pdb')
    clean_pdb(protein_file,protein_nowat_file)
    #fpocket
    os.system('fpocket -f '+protein_nowat_file)
    fpocket_dir=os.path.join(protein_nowat_file.replace('.pdb','_out'),'pockets')
    get_centers(fpocket_dir)
    barycenter_file=os.path.join(fpocket_dir,'bary_centers.txt')
    #types and gninatyper
    protein_gninatype=gninatype(protein_nowat_file)
    class_types=create_types(barycenter_file,protein_gninatype)
    #rank pockets
    class_model=Model()
    class_checkpoint=torch.load(args.class_checkpoint)
    types_lines=open(class_types,'r').readlines()
    batch_size = len(types_lines)
    #avoid cuda out of memory
    if batch_size>50:
        batch_size=50
    class_model, class_gmaker, class_eptest=get_model_gmaker_eprovider(class_types,batch_size,class_model,class_checkpoint)
    #divisible by 50 if types_lines > 50
    class_labels, class_probs = test_model(class_model, class_eptest, class_gmaker,  batch_size)
    zipped_lists = zip(class_probs[:len(types_lines)], types_lines)
    sorted_zipped_lists = sorted(zipped_lists,reverse=True)
    ranked_types = [element for _, element in sorted_zipped_lists]
    seg_types= class_types.replace('.types','_ranked.types')
    fout=open(seg_types,'w')
    fout.write(''.join(ranked_types))
    fout.close()
    del class_model
    del class_checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    #segmentation
    if args.rank!=0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seg_model = Unet(args.num_classes, args.upsample)
        seg_model.to(device)
        seg_checkpoint = torch.load(args.seg_checkpoint)
        seg_model = nn.DataParallel(seg_model)
        seg_model, seg_gmaker, seg_eptest=get_model_gmaker_eprovider(seg_types,1,seg_model,seg_checkpoint,dims=32)
        dx_name=protein_nowat_file.replace('.pdb','')
        test(seg_model, seg_eptest, seg_gmaker,device,dx_name, args)
