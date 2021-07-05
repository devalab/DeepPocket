'''Benchmark DeepPocket segmentation model on Zhao. et. al. benchmark. Prints out IOUs and success rates of ratio thresholds for different distances and ratio thresholds'''
from prody import *
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
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import AllChem

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
    eptest = molgrid.ExampleProvider(shuffle=False, stratify_receptor=False,iteration_scheme=molgrid.IterationScheme.LargeEpoch,default_batch_size=1,data_root=args.data_dir,recmolcache=args.test_recmolcache)
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

def binding_site_AA(ligand,prot_prody,distance):
    #amino acids from ligand distance threshold
    prot_coords = prot_prody.getCoords()
    c = ligand.GetConformer()
    ligand_coords = c.GetPositions()
    ligand_dist = cdist(ligand_coords, prot_coords)
    binding_indices = np.where(np.any(ligand_dist <= distance, axis=0))
    #Get protein residue indices involved in binding site
    prot_resin = prot_prody.getResindices()
    prot_binding_indices = prot_resin[binding_indices]
    prot_binding_indices = sorted(list(set(prot_binding_indices)))
    return prot_binding_indices

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

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def union(lst1, lst2):
    return list(set().union(lst1,lst2))

def parse_args(argv=None):
    '''Return argument namespace and commandline'''
    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('--test_types', type=str, required=True,
                        help="test types file")
    parser.add_argument('--model_weights', type=str, required=True,
                        help="weights for UNET")
    parser.add_argument('-t', '--threshold', type=float, required=False,
                        help="threshold for segmentation", default=0.5)
    parser.add_argument('--upsample', type=str, required=False,
                        help="Type of Upsampling", default=None)
    parser.add_argument('--num_classes', type=int, required=False,
                        help="Output channels for predicted masks, default 1", default=1)
    parser.add_argument('-d', '--data_dir', type=str, required=False,
                        help="Root directory of data", default="")
    parser.add_argument('--test_recmolcache', type=str, required=False,
                        help="path to test receptor molcache", default="")
    args = parser.parse_args(argv)

    argdict = vars(args)
    line = ''
    for (name, val) in list(argdict.items()):
        if val != parser.get_default(name):
            line += ' --%s=%s' % (name, val)

    return (args, line)

def test(model, test_loader, gmaker_img,device, args,ligand_distances,mask_distances,ratios,count_values,IOUS):
    with torch.no_grad():
        count=0
        model.eval()
        dims = gmaker_img.grid_dimensions(test_loader.num_types())
        tensor_shape = (1,) + dims
        #create tensor for input, centers and indices
        input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device=device, requires_grad=True)
        float_labels = torch.zeros((1, 4), dtype=torch.float32, device=device)
        for batch in test_loader:
            # update float_labels with center and index values
            batch.extract_labels(float_labels)
            centers = float_labels[:, 1:]
            for b in range(1):
                #get protein and ligand files
                protein_file=os.path.join(args.data_dir,batch[b].coord_sets[0].src.replace('.gninatypes','.pdb'))
                ligand_file=os.path.join(args.data_dir,batch[b].coord_sets[0].src.replace('protein_nowat.gninatypes','ligand.sdf'))
                #load in protein and ligand
                ligand=Chem.MolFromMolFile(ligand_file,sanitize=False)
                prot_prody=parsePDB(protein_file)
                center = molgrid.float3(float(centers[b][0]), float(centers[b][1]), float(centers[b][2]))
                # Update input tensor with b'th datapoint of the batch
                gmaker_img.forward(center, batch[b].coord_sets[0], input_tensor[b])
            # Take only the first 14 channels as that is for proteins, other 14 are ligands and will remain 0.
            masks_pred = model(input_tensor[:, :14])
            masks_pred=masks_pred.detach().cpu()
            masks_pred=preprocess_output(masks_pred[0], args.threshold)
            pred_coords = Output_Coordinates(masks_pred, center)
            for ld in range(len(ligand_distances)):
                true_aa = binding_site_AA(ligand, prot_prody, ligand_distances[ld])
                for md in range(len(mask_distances)):
                    pred_aa = predicted_AA(pred_coords, prot_prody, mask_distances[md])
                    intersect = intersection(pred_aa, true_aa)
                    un = union(pred_aa, true_aa)
                    IOUS[ld][md]+=len(intersect)/len(un)
                    for r in range(len(ratios)):
                        if len(intersect)/len(true_aa)>=ratios[r]:
                            count_values[ld][r][md]+=1    
    return count_values
if __name__ == "__main__":
    ligand_distances=[3,4,5]
    ratios=[0.25,0.5,0.75]
    mask_distances=[1,1.5,2,2.5,3,3.5]
    count_values=np.zeros((len(ligand_distances),len(ratios),len(mask_distances)))
    IOUS=np.zeros((len(ligand_distances),len(mask_distances)))
    (args, cmdline) = parse_args()
    gmaker_img, eptest = get_model_gmaker_eproviders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(args.num_classes, args.upsample)
    model.to(device)
    checkpoint = torch.load(args.model_weights)
    model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    count_values=test(model, eptest, gmaker_img,device,args,ligand_distances,mask_distances,ratios,count_values,IOUS)
    count_values/=4414
    IOUS/=4414
    for ld in range(len(ligand_distances)):
        for md in range(len(mask_distances)):
            print("ligand distance ", ligand_distances[ld], "mask_distance ", mask_distances[md], "IOU ", IOUS[ld][md])
            for r in range(len(ratios)):
                print("ligand distance ", ligand_distances[ld], "mask_distance ", mask_distances[md], "ratio ", ratios[r], "value ",count_values[ld][r][md])

