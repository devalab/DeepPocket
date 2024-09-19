'''
Segment out pocket shapes from top ranked pockets
'''
import torch
import torch.nn as nn
from unet import Unet
import numpy as np
import logging
import argparse
import wandb
import sys
import os
import molgrid
from skimage.morphology import binary_dilation
from skimage.morphology import cube
from skimage.morphology import closing
from skimage.segmentation import clear_border
from skimage.measure import label
from scipy.spatial.distance import cdist
from prody import *

def preprocess_output(input, threshold):
    input[input>=threshold]=1
    input[input!=1]=0
    input=input.numpy()
    bw = closing(input).any(axis=0)
    # remove artifacts connected to border
    cleared = clear_border(bw)

    # label regions
    label_image, num_labels = label(cleared, return_num=True)
    largest=0
    for i in range(1, num_labels + 1):
        pocket_idx = (label_image == i)
        pocket_size = pocket_idx.sum()
        if pocket_size >largest:
            largest=pocket_size
    for i in range(1, num_labels + 1):
        pocket_idx = (label_image == i)
        pocket_size = pocket_idx.sum()
        if pocket_size <largest:
            label_image[np.where(pocket_idx)] = 0
    label_image[label_image>0]=1
    return torch.tensor(label_image,dtype=torch.float32)

def get_model_gmaker_eproviders(args):
    # test example provider
    eptest = molgrid.ExampleProvider(shuffle=False, stratify_receptor=False,iteration_scheme=molgrid.IterationScheme.LargeEpoch,default_batch_size=1)
    eptest.populate(args.test_types)
    # gridmaker with defaults
    gmaker_img = molgrid.GridMaker(dimension=32)

    return  gmaker_img, eptest

def Output_Coordinates(tensor,center,dimension=16.25,resolution=0.5):
    #get coordinates of mask from predicted mask
    tensor=tensor.numpy()
    indices = np.argwhere(tensor>0).astype('float32')
    indices *= resolution
    center=np.array([float(center[0]), float(center[1]), float(center[2])])
    indices += center
    indices -= dimension
    return indices

def predicted_AA(indices,prot_prody,distance):
    #amino acids from mask distance thresholds
    prot_coords = prot_prody.getCoords()
    ligand_dist = cdist(indices, prot_coords)
    binding_indices = np.where(np.any(ligand_dist <= distance, axis=0))
    #get predicted protein residue indices involved in binding site
    prot_resin = prot_prody.getResindices()
    prot_binding_indices = prot_resin[binding_indices]
    prot_binding_indices = sorted(list(set(prot_binding_indices)))
    return prot_binding_indices

def output_pocket_pdb(pocket_name,prot_prody,pred_AA):
    #output pocket pdb
    #skip if no amino acids predicted
    if len(pred_AA)==0:
        return
    sel_str= 'resindex '
    for i in pred_AA:
        sel_str+= str(i)+' or resindex '
    sel_str=' '.join(sel_str.split()[:-2])
    pocket=prot_prody.select(sel_str)
    writePDB(pocket_name,pocket)

def parse_args(argv=None):
    '''Return argument namespace and commandline'''
    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('--test_types', type=str, required=True,
                        help="test types file")
    parser.add_argument('--model_weights', type=str, required=True,
                        help="weights for UNET")
    parser.add_argument('-t', '--threshold', type=float, required=False,
                        help="threshold for segmentation", default=0.5)
    parser.add_argument('-r', '--rank', type=int, required=False,
                        help="number of pockets to segment", default=1)
    parser.add_argument('--upsample', type=str, required=False,
                        help="Type of Upsampling", default=None)
    parser.add_argument('--num_classes', type=int, required=False,
                        help="Output channels for predicted masks, default 1", default=1)
    parser.add_argument('--dx_name', type=str, required=True,
                        help="dx file name")
    parser.add_argument('-p','--protein', type=str, required=False, help="pdb file for predicting binding sites")
    parser.add_argument('--mask_dist', type=float, required=False,
                        help="distance from mask to residues", default=3.5)
    args = parser.parse_args(argv)

    argdict = vars(args)
    line = ''
    for (name, val) in list(argdict.items()):
        if val != parser.get_default(name):
            line += ' --%s=%s' % (name, val)

    return (args, line)

def test(model, test_loader, gmaker_img,device,dx_name, args):
    if args.rank==0:
        return
    count=0
    model.eval()
    dims = gmaker_img.grid_dimensions(test_loader.num_types())
    tensor_shape = (1,) + dims
    #create tensor for input, centers and indices
    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device=device, requires_grad=True)
    float_labels = torch.zeros((1, 4), dtype=torch.float32, device=device)
    prot_prody=parsePDB(args.protein)
    for batch in test_loader:
        count+=1
        # update float_labels with center and index values
        batch.extract_labels(float_labels)
        centers = float_labels[:, 1:]
        for b in range(1):
            center = molgrid.float3(float(centers[b][0]), float(centers[b][1]), float(centers[b][2]))
            # Update input tensor with b'th datapoint of the batch 
            gmaker_img.forward(center, batch[b].coord_sets[0], input_tensor[b])
        # Take only the first 14 channels as that is for proteins, other 14 are ligands and will remain 0.
        masks_pred = model(input_tensor[:, :14])
        masks_pred=masks_pred.detach().cpu()
        masks_pred=preprocess_output(masks_pred[0], args.threshold)
        # predict binding site residues
        pred_coords = Output_Coordinates(masks_pred, center)
        pred_aa = predicted_AA(pred_coords, prot_prody, args.mask_dist)
        output_pocket_pdb(dx_name+'_pocket'+str(count)+'.pdb',prot_prody,pred_aa)
        masks_pred=masks_pred.cpu()
        # Output predicted mask in .dx format 
        masks_pred=molgrid.Grid3f(masks_pred)
        molgrid.write_dx(dx_name+'_'+str(count)+'.dx',masks_pred,center,0.5,1.0)
        if count>=args.rank:
            break

if __name__ == "__main__":
    (args, cmdline) = parse_args()
    gmaker_img, eptest = get_model_gmaker_eproviders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(args.num_classes, args.upsample)
    model.to(device)
    checkpoint = torch.load(args.model_weights)
    model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    dx_name=args.dx_name
    test(model, eptest, gmaker_img,device,dx_name, args)
