'''
Rank candidate pockets provided by fpocket using the classification model
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import sys
import imp
import molgrid
import argparse
import os
import time
torch.backends.cudnn.benchmark = True

def parse_args(argv=None):
    '''Return argument namespace and commandline'''
    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Model template python file")
    parser.add_argument('--test_types', type=str, required=True,
                        help="test types file")
    parser.add_argument('--checkpoint', type=str, required=False, help="file to continue training from")
    args = parser.parse_args(argv)

    argdict = vars(args)
    line = ''
    for (name, val) in list(argdict.items()):
        if val != parser.get_default(name):
            line += ' --%s=%s' % (name, val)

    return (args, line)

def initialize_model(model, args):
    checkpoint = torch.load(args.checkpoint)
    model.cuda()
    model.load_state_dict(checkpoint['model_state_dict'])

def get_model_gmaker_eproviders(args,batch_size):
    # test example provider
    eptest_large = molgrid.ExampleProvider(shuffle=False, stratify_receptor=False, labelpos=0,balanced=False,
                                     iteration_scheme=molgrid.IterationScheme.LargeEpoch,default_batch_size=batch_size)
    eptest_large.populate(args.test_types)
    eptest_small = molgrid.ExampleProvider(shuffle=True, stratify_receptor=True, labelpos=0, balanced=True,
                                           iteration_scheme=molgrid.IterationScheme.SmallEpoch,
                                           default_batch_size=batch_size)
    eptest_small.populate(args.test_types)
    # gridmaker with defaults
    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(eptest_small.num_types())
    model_file = imp.load_source("model", args.model)
    # load model with seed
    model = model_file.Model()

    return model, gmaker,  eptest_large,eptest_small

def test_model(model, ep, gmaker,  batch_size):
        t=time.time()
        # loss accumulation
        all_labels=[]
        all_probs=[]
        # testing setup
        # testing loop
        dims = gmaker.grid_dimensions(ep.num_types())
        tensor_shape = (batch_size,) + dims
        criterion = nn.CrossEntropyLoss()
        #create tensor for input, center and index
        input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda', requires_grad=True)
        float_labels = torch.zeros((batch_size,4), dtype=torch.float32, device='cuda')
        count=0
        for batch in ep:
            count+=1
            # update float_labels with center and index values
            batch.extract_labels(float_labels)
            centers = float_labels[:,1:]
            labels = float_labels[:,0].long().to('cuda')
            for b in range(batch_size):
                center = molgrid.float3(float(centers[b][0]),float(centers[b][1]),float(centers[b][2]))
                # Update input tensor with b'th datapoint of the batch 
                gmaker.forward(center,batch[b].coord_sets[0],input_tensor[b])
            # Take only the first 14 channels as that is for proteins, other 14 are ligands and will remain 0.
            output = model(input_tensor[:,:14])
            #labels_oh = nn.functional.one_hot(labels)
            #labels_oh = labels_oh
            all_labels.append(labels.cpu())
            all_probs.append(F.softmax(output).detach().cpu())
        all_labels=torch.flatten(torch.stack(all_labels)).cpu().numpy()
        #all predicted probabilities
        all_probs=torch.flatten(torch.stack(all_probs),start_dim=0,end_dim=1).cpu().numpy()
        # saving cuda memory
        del input_tensor
        return all_labels, all_probs[:,1]

if __name__ == '__main__':
    (args, cmdline) = parse_args()
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
    types_lines=open(args.test_types,'r').readlines()
    batch_size = len(types_lines)
    model, gmaker,  eptest_large,eptest_small = get_model_gmaker_eproviders(args,batch_size)
    initialize_model(model, args)
    all_labels, all_probs = test_model(model, eptest_large, gmaker,  batch_size)
    zipped_lists = zip(all_probs, types_lines)
    sorted_zipped_lists = sorted(zipped_lists,reverse=True)
    ranked_types = [element for _, element in sorted_zipped_lists]
    fout=open(args.test_types.replace('.types','_ranked.types'),'w')
    fout.write(''.join(ranked_types))
    fout.close()
