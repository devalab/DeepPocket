import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from unet import Unet
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


import matplotlib.pyplot as plt
import numpy as np
import logging
import argparse
import wandb
import sys
import os
import molgrid
from skimage.morphology import binary_dilation
from skimage.morphology import cube
torch.backends.cudnn.benchmark = True

def get_mask(coordinateset,center,gmaker):
    # Create ground truth tensor
    c2grid = molgrid.Coords2Grid(gmaker, center=center)
    origtypes = torch.ones(coordinateset.coords.tonumpy().shape[0], 1)
    radii = torch.ones((coordinateset.coords.tonumpy().shape[0]))
    grid_gen = c2grid(torch.tensor(coordinateset.coords.tonumpy()), origtypes, radii)
    grid_np = grid_gen.numpy()
    grid_np=binary_dilation(grid_np[0],cube(3))
    grid_np =grid_np.astype(float)
    return torch.tensor(np.expand_dims(grid_np,axis=0))

def parse_args(argv=None):
    '''Return argument namespace and commandline'''
    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('--train_types', type=str, required=True,
                        help="training types file")
    parser.add_argument('--upsample', type=str, required=False,
                        help="Type of Upsampling", default=None)
    parser.add_argument('--test_types', type=str, required=True,
                        help="test types file")
    parser.add_argument('-d', '--data_dir', type=str, required=False,
                        help="Root directory of data", default="")
    parser.add_argument('--train_recmolcache', type=str, required=False,
                        help="path to train receptor molcache", default="")
    parser.add_argument('--test_recmolcache', type=str, required=False,
                        help="path to test receptor molcache", default="")
    parser.add_argument('-e', '--num_epochs', type=int, required=False,
                        help="Number of epochs", default=50)
    parser.add_argument('-b', '--batch_size', type=int, required=False,
                        help="Batch size for training, default 50", default=40)
    parser.add_argument('--num_classes', type=int, required=False,
                        help="Output channels for predicted masks, default 1", default=1)
    parser.add_argument('-s', '--seed', type=int, required=False, help="Random seed, default 0", default=0)
    parser.add_argument('-r', '--run_name', type=str, help="name for wandb run", required=False)
    parser.add_argument('-o', '--outprefix', type=str, help="Prefix for output files", required=True)
    parser.add_argument('--checkpoint', type=str, required=False, help="file to continue training from")
    parser.add_argument('--solver', type=str, help="Solver type. Default is SGD, Nesterov or Adam", default='SGD')
    parser.add_argument('--step_reduce', type=float,
                        help="Reduce the learning rate by this factor with dynamic stepping, default 0.1",
                        default=0.1)
    parser.add_argument('--step_end_cnt', type=float, help='Terminate training after this many lr reductions',
                        default=3)
    parser.add_argument('--step_when', type=int,
                        help="Perform a dynamic step (reduce base_lr) when training has not improved after these many epochs, default 2",
                        default=15)
    parser.add_argument('--base_lr', type=float, help='Initial learning rate, default 0.01', default=0.01)
    parser.add_argument('--momentum', type=float, help="Momentum parameters, default 0.9", default=0.9)
    parser.add_argument('--weight_decay', type=float, help="Weight decay, default 0.001", default=0.001)
    parser.add_argument('--clip_gradients', type=float, default=10.0, help="Clip gradients threshold (default 10)")
    args = parser.parse_args(argv)

    argdict = vars(args)
    line = ''
    for (name, val) in list(argdict.items()):
        if val != parser.get_default(name):
            line += ' --%s=%s' % (name, val)

    return (args, line)


def cal_dice_coeff(input, target):
    eps = 0.0001
    inter = torch.dot(input.view(-1), target.view(-1))
    union = torch.sum(input) + torch.sum(target) + eps
    t = (2 * inter.float() + eps) / union.float()
    return t

def cal_IOU(input, target):
    inter = torch.dot(input.view(-1), target.view(-1))
    union = torch.sum(input) + torch.sum(target)
    t = (inter.float()) / (union.float() - inter.float())
    return t


def get_model_gmaker_eproviders(args):
    # train example provider
    eptrain = molgrid.ExampleProvider(shuffle=True, stratify_receptor=False,balanced=False,data_root=args.data_dir,recmolcache=args.train_recmolcache,iteration_scheme=molgrid.IterationScheme.LargeEpoch,default_batch_size=args.batch_size,cache_structs=True)
    eptrain.populate(args.train_types)
    print(round(eptrain.large_epoch_size()/args.batch_size))
    # test example provider
    eptest = molgrid.ExampleProvider(shuffle=False, stratify_receptor=False,data_root=args.data_dir,iteration_scheme=molgrid.IterationScheme.LargeEpoch,default_batch_size=args.batch_size,recmolcache=args.test_recmolcache,balanced=False,cache_structs=True)
    eptest.populate(args.test_types)
    print(round(eptest.large_epoch_size()/args.batch_size))
    # gridmaker with defaults
    gmaker_img = molgrid.GridMaker(dimension=32)
    dims = gmaker_img.grid_dimensions(eptrain.num_types())
    #grid maker for ground truth tensor
    gmaker_mask = molgrid.GridMaker(dimension=32,binary=True,gaussian_radius_multiple=-1,resolution=0.5)

    return  gmaker_img, gmaker_mask,eptrain, eptest


def initialize_model(model, args):
    def weights_init(m):
        '''initialize model weights with xavier'''
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            torch.nn.init.kaiming_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

        if isinstance(m, nn.BatchNorm3d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.cuda()
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.apply(weights_init)


def train(model, train_loader, test_loader,gmaker_img,gmaker_mask, args, device):
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
    initialize_model(model, args)
    wandb.watch(model)
    num_epochs = args.num_epochs
    outprefix = args.outprefix
    prev_total_loss_snap = ''
    prev_total_accuracy_snap = ''
    prev_total_dice_snap = ''
    prev_total_IOU_snap = ''
    prev_snap = ''
    initial = 0
    # global_step = 0
    box_size = 65
    last_test = 0

    if args.checkpoint:
        initial = checkpoint['Epoch']

    if 'SGD' in args.solver:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif 'Nesterov' in args.solver:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
    elif 'Adam' in args.solver:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    else:
        print("No test solver argument passed (SGD, Adam, Nesterov)")
        sys.exit(1)

    if args.checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=args.step_reduce,
                                                           patience=args.step_when, verbose=True)
    if args.checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    Bests = {}
    Bests['train_epoch'] = 0
    Bests['test_loss'] = torch.from_numpy(np.asarray(np.inf))
    Bests['test_accuracy'] = torch.from_numpy(np.asarray([0]))
    Bests['dice_coeff'] = torch.from_numpy(np.asarray([0]))
    Bests['IOU'] = torch.from_numpy(np.asarray([0]))
    if args.checkpoint:
        Bests = checkpoint['Bests']
    criterion = nn.BCEWithLogitsLoss()
    dims = gmaker_img.grid_dimensions(eptrain.num_types())
    tensor_shape = (args.batch_size,) + dims
    mask_shape=(args.batch_size,1) + dims[1:]
    #create tensor for input, ground truth/mask, centers and labels
    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device=device, requires_grad=True)
    mask_tensor = torch.empty(mask_shape, dtype=torch.float32, device=device, requires_grad=True)
    float_labels = torch.zeros((args.batch_size, 4), dtype=torch.float32, device=device)
    logging.info("Started Training.....")
    for epoch in range(initial, num_epochs):
        model.train()
        #running_acc = 0.0
        #running_loss = 0.0
        for batch in train_loader:
            # extract labels and centers of batch datapoints
            batch.extract_labels(float_labels)
            centers = float_labels[:, 1:]
            for b in range(args.batch_size):
                center = molgrid.float3(float(centers[b][0]), float(centers[b][1]), float(centers[b][2]))
                #intialise transformer for rotaional augmentation
                transformer = molgrid.Transform(center, 0, True)
                # random rotation on input protein
                transformer.forward(batch[b], batch[b])
                # Update input tensor with b'th datapoint of the batch
                gmaker_img.forward(center, batch[b].coord_sets[0], input_tensor[b])
                with torch.no_grad():
                    # Update mask tensor with b'th datapoint ground truth of the batch
                    mask_tensor[b]=get_mask(batch[b].coord_sets[-1],center,gmaker_mask).to(device)
            optimizer.zero_grad()
            # Take only the first 14 channels as that is for proteins, other 14 are ligands and will remain 0.
            masks_pred = model(input_tensor[:,:14])
            loss = criterion(masks_pred, mask_tensor)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradients)
            optimizer.step()
            #running_loss += loss.item()
            _, predictions = torch.max(masks_pred, 1)
            acc=torch.mean(
                (mask_tensor == predictions).float())# Pixel-wise accuracy can be misleading in case of class imbalance
            #running_acc += acc
            pred = torch.sigmoid(masks_pred)
            pred = (pred > 0.5).float()
            dice= cal_dice_coeff(pred, mask_tensor)
            IOU = cal_IOU(pred,mask_tensor)
            wandb.log({'train_loss': loss.item(), 'train_accuracy': acc,'train_dice':dice, 'train_IOU': IOU })

        #train_acc = running_acc / (round(train_loader.large_epoch_size()/args.batch_size))
        #train_loss = running_loss / (round(train_loader.large_epoch_size()/args.batch_size))
        #print("epoch "+ epoch +" train_acc "+ train_acc + " train_loss " + train_loss)

        test_loss, test_acc, dice_coeff,IOU = test(model, test_loader,gmaker_img ,gmaker_mask, args, criterion,device)

        scheduler.step(dice_coeff)
        if test_loss < Bests['test_loss']:
            Bests['test_loss'] = test_loss
            wandb.run.summary["test_loss"] = Bests['test_loss']
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'Bests': Bests,
                        'Epoch': epoch + 1}, outprefix + '_best_test_loss_' + str(epoch + 1) + '.pth.tar')
            if prev_total_loss_snap:
                os.remove(prev_total_loss_snap)
            prev_total_loss_snap = outprefix + '_best_test_loss_' + str(epoch + 1) + '.pth.tar'
        if test_acc > Bests['test_accuracy']:
            Bests['test_accuracy'] = test_acc
            wandb.run.summary["test_accuracy"] = Bests['test_accuracy']
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'Bests': Bests,
                        'Epoch': epoch + 1}, outprefix + '_best_test_accuracy_' + str(epoch + 1) + '.pth.tar')
            if prev_total_accuracy_snap:
                os.remove(prev_total_accuracy_snap)
            prev_total_accuracy_snap = outprefix + '_best_test_accuracy_' + str(epoch + 1) + '.pth.tar'
        if IOU > Bests['IOU']:
            Bests['IOU'] = IOU
            wandb.run.summary["IOU"] = Bests['IOU']
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'Bests': Bests,
                        'Epoch': epoch + 1}, outprefix + '_best_test_IOU_' + str(epoch + 1) + '.pth.tar')
            if prev_total_IOU_snap:
                os.remove(prev_total_IOU_snap)
            prev_total_IOU_snap = outprefix + '_best_test_IOU_' + str(epoch + 1) + '.pth.tar'
        if dice_coeff > Bests['dice_coeff']:
            Bests['dice_coeff'] = dice_coeff
            wandb.run.summary["dice_coeff"] = Bests['dice_coeff']
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'Bests': Bests,
                        'Epoch': epoch + 1}, outprefix + '_best_dice_coeff_' + str(epoch + 1) + '.pth.tar')
            if prev_total_dice_snap:
                os.remove(prev_total_dice_snap)
            prev_total_dice_snap = outprefix + '_best_dice_coeff_' + str(epoch + 1) + '.pth.tar'
            Bests['train_epoch'] = epoch
        if epoch - Bests['train_epoch'] >= args.step_when and optimizer.param_groups[0]['lr'] <= (
                    (args.step_reduce) ** args.step_end_cnt) * args.base_lr:
            last_test = 1
        print(
                "Epoch {}, total_test_loss: {:.3f},total_test_accuracy: {:.3f},total_dice_coeff: {:.3f}, Best_test_loss: {:.3f},Best_test_accuracy: {:.3f},Best_dice_coeff: {:.3f},learning_Rate: {:.7f}".format(
                    epoch + 1, test_loss, test_acc, dice_coeff, Bests['test_loss'], Bests['test_accuracy'],
                    Bests['dice_coeff'], optimizer.param_groups[0]['lr']))
        wandb.log({'test_loss': test_loss, 'test_accuracy': test_acc, 'dice_coeff': dice_coeff,
                       'learning rate': optimizer.param_groups[0]['lr']})
        torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'Bests': Bests,
                        'Epoch': epoch + 1}, outprefix + '_' + str(epoch + 1) + '.pth.tar')
        if prev_snap:
            os.remove(prev_snap)
        prev_snap = outprefix + '_' + str(epoch + 1) + '.pth.tar'
        if last_test:
            return Bests

    logging.info("Finished Training")


def test(model, test_loader, gmaker_img,gmaker_mask, args, criterion,device):
    model.eval()

    running_acc = 0.0
    running_loss = 0.0
    tot_dice = 0.0
    tot_IOU=0.0
    dice_coeff = None
    dims = gmaker_img.grid_dimensions(eptrain.num_types())
    tensor_shape = (args.batch_size,) + dims
    mask_shape = (args.batch_size, 1) + dims[1:]
    #create tensor for input, ground truth/mask, centers and labels
    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device=device, requires_grad=True)
    mask_tensor = torch.empty(mask_shape, dtype=torch.float32, device=device, requires_grad=True)
    float_labels = torch.zeros((args.batch_size, 4), dtype=torch.float32, device=device)

    for batch in test_loader:
        # extract labels and centers of batch datapoints
        batch.extract_labels(float_labels)
        centers = float_labels[:, 1:]
        for b in range(args.batch_size):
            center = molgrid.float3(float(centers[b][0]), float(centers[b][1]), float(centers[b][2]))
            transformer = molgrid.Transform(center, 0, True)
            transformer.forward(batch[b], batch[b])
            # Update input tensor with b'th datapoint of the batch
            gmaker_img.forward(center, batch[b].coord_sets[0], input_tensor[b])
            with torch.no_grad():
                # Update mask tensor with b'th datapoint ground truth of the batch
                mask_tensor[b] = get_mask(batch[b].coord_sets[-1], center, gmaker_mask).to(device)
        # Take only the first 14 channels as that is for proteins, other 14 are ligands and will remain 0.
        masks_pred = model(input_tensor[:,:14])
        loss = criterion(masks_pred, mask_tensor)
        _, predictions = torch.max(masks_pred, 1)
        running_loss += loss.detach().cpu()
        running_acc += torch.mean((mask_tensor == predictions).float()).detach().cpu()
        pred = torch.sigmoid(masks_pred)
        pred = (pred > 0.5).float()

        tot_dice += cal_dice_coeff(pred, mask_tensor).detach().cpu()
        tot_IOU += cal_IOU(pred,mask_tensor).detach().cpu()
    test_loss = running_loss / (round(test_loader.large_epoch_size()/args.batch_size))
    test_acc = running_acc / (round(test_loader.large_epoch_size()/args.batch_size))

    dice_coeff = tot_dice / (round(test_loader.large_epoch_size()/args.batch_size))
    IOU = tot_IOU/(round(test_loader.large_epoch_size()/args.batch_size))
    return test_loss, test_acc, dice_coeff,IOU


if __name__ == "__main__":
    (args, cmdline) = parse_args()
    wandb.init(project='deep-pocket', name=args.run_name)
    gmaker_img, gmaker_mask,eptrain, eptest=get_model_gmaker_eproviders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(args.num_classes, args.upsample)
    model.to(device)
    model=nn.DataParallel(model)
    Bests = train(model, eptrain, eptest,gmaker_img,gmaker_mask, args, device)
    print(Bests)
