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
import wandb
from sklearn.metrics import precision_recall_fscore_support

torch.backends.cudnn.benchmark = True
def parse_args(argv=None):
    '''Return argument namespace and commandline'''
    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Model template python file")
    parser.add_argument('--train_types', type=str, required=True,
                        help="training types file")
    parser.add_argument('--test_types', type=str, required=True,
                        help="test types file")
    parser.add_argument('-i', '--iterations', type=int, required=False,
                        help="Number of iterations to run,default 10,000", default=10000)
    parser.add_argument('-d', '--data_dir', type=str, required=False,
                        help="Root directory of data", default="")
    parser.add_argument('--train_recmolcache', type=str, required=False,
                        help="path to receptor molcache", default="")
    parser.add_argument('--test_recmolcache', type=str, required=False,
                        help="path to receptor molcache", default="")
    parser.add_argument('-b', '--batch_size', type=int, required=False,
                        help="Batch size for training, default 50", default=50)
    parser.add_argument('-s', '--seed', type=int, required=False, help="Random seed, default 0", default=0)
    parser.add_argument('-t', '--test_interval', type=int, help="How frequently to test (iterations), default 1000",
                        default=1000)
    parser.add_argument('-r', '--run_name', type=str, help="name for wandb run", required=False)
    parser.add_argument('-o', '--outprefix', type=str, help="Prefix for output files", required=True)
    parser.add_argument('--percent_reduced', type=float, default=100,
                        help='Create a reduced set on the fly based on types file, using the given percentage: to use 10 percent pass 10. Range (0,100)')
    parser.add_argument('--checkpoint', type=str, required=False, help="file to continue training from")
    parser.add_argument('--solver', type=str, help="Solver type. Default is SGD, Nesterov or Adam", default='SGD')
    parser.add_argument('--step_reduce', type=float,
                        help="Reduce the learning rate by this factor with dynamic stepping, default 0.1",
                        default=0.1)
    parser.add_argument('--step_end_cnt', type=float, help='Terminate training after this many lr reductions',
                        default=3)
    parser.add_argument('--step_when', type=int,
                        help="Perform a dynamic step (reduce base_lr) when training has not improved after this many test iterations, default 15",
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


def initialize_model(model, args):
    def weights_init(m):
        '''initialize model weights with xavier'''
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.cuda()
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.apply(weights_init)


def get_model_gmaker_eproviders(args):
    # train example provider
    eptrain = molgrid.ExampleProvider(shuffle=True, stratify_receptor=True, labelpos=0,balanced=True,
                                      data_root=args.data_dir,recmolcache=args.train_recmolcache)
    eptrain.populate(args.train_types)
    # test example provider
    eptest_large = molgrid.ExampleProvider(shuffle=False, stratify_receptor=False, labelpos=0,balanced=False,
                                     data_root=args.data_dir,iteration_scheme=molgrid.IterationScheme.LargeEpoch,default_batch_size=args.batch_size,recmolcache=args.test_recmolcache)
    eptest_large.populate(args.test_types)
    eptest_small = molgrid.ExampleProvider(shuffle=True, stratify_receptor=True, labelpos=0, balanced=True,
                                           data_root=args.data_dir, iteration_scheme=molgrid.IterationScheme.SmallEpoch,
                                           default_batch_size=args.batch_size,recmolcache=args.test_recmolcache)
    eptest_small.populate(args.test_types)
    # gridmaker with defaults
    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(eptrain.num_types())
    model_file = imp.load_source("model", args.model)
    # load model with seed
    torch.manual_seed(args.seed)
    model = model_file.Model()

    return model, gmaker, eptrain, eptest_large,eptest_small

def train_and_test(args, model, eptrain, eptest_large,eptest_small, gmaker):
    def test_model(model, ep, gmaker, percent_reduced, batch_size):
        t=time.time()
        # loss accumulation
        all_losses=[]
        all_accuracy = []
        all_labels=[]
        all_probs=[]
        # testing setup
        # testing loop
        criterion = nn.CrossEntropyLoss()
        # Create tensors for input, center and labels
        input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda', requires_grad=True)
        float_labels = torch.zeros((batch_size,4), dtype=torch.float32, device='cuda')
        count=0
        for batch in ep:
            count+=1
            # extract labels and centers of batch datapoints
            batch.extract_labels(float_labels)
            centers = float_labels[:,1:]
            labels = float_labels[:,0].long().to('cuda')
            for b in range(batch_size):
                center = molgrid.float3(float(centers[b][0]),float(centers[b][1]),float(centers[b][2]))
                # Update input tensor with b'th datapoint of the batch
                gmaker.forward(center,batch[b].coord_sets[0],input_tensor[b])
            # Take only the first 14 channels as that is for proteins, other 14 are for ligand and will remain 0.
            output = model(input_tensor[:,:14])
            #labels_oh = nn.functional.one_hot(labels)
            #labels_oh = labels_oh
            loss = criterion(output,labels)
            predicted=torch.argmax(output,dim=1)
            accuracy= labels.eq(predicted).sum().float()/batch_size
            all_losses.append(loss.detach())
            all_accuracy.append(accuracy)
            all_labels.append(labels.cpu())
            all_probs.append(F.softmax(output).detach().cpu())
        # mean loss for testing session
        all_labels=torch.flatten(torch.stack(all_labels)).cpu().numpy()
        all_probs=torch.flatten(torch.stack(all_probs),start_dim=0,end_dim=1).cpu().numpy()
        total_test_loss_mean = torch.mean(torch.stack(all_losses)).cpu()
        total_test_accuracy_mean = torch.mean(torch.stack(all_accuracy)).cpu()
        auc = roc_auc_score(all_labels, all_probs[:,1])
        return total_test_loss_mean, total_test_accuracy_mean, auc, all_labels, all_probs

    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
    initialize_model(model, args)
    wandb.watch(model)
    iterations = args.iterations
    test_interval = args.test_interval
    batch_size = args.batch_size
    percent_reduced = args.percent_reduced
    outprefix = args.outprefix
    prev_total_loss_snap = ''
    prev_total_accuracy_snap = ''
    prev_total_auc_snap = ''
    prev_snap = ''
    initial = 0
    if args.checkpoint:
        initial = checkpoint['Iteration']
    last_test = 0

    if 'SGD' in args.solver:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif 'Nesterov' in args.solver:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
    elif 'Adam' in args.solver:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    else:
        print("No valid solver argument passed (SGD, Adam, Nesterov)")
        sys.exit(1)
    if args.checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=args.step_reduce,
                                                           patience=args.step_when, verbose=True)
    if args.checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    Bests = {}
    Bests['train_iteration'] = 0
    Bests['test_loss'] = torch.from_numpy(np.asarray(np.inf))
    Bests['test_accuracy'] = torch.from_numpy(np.asarray([0]))
    Bests['test_auc'] = torch.from_numpy(np.asarray([0]))
    if args.checkpoint:
        Bests = checkpoint['Bests']

    dims = gmaker.grid_dimensions(eptrain.num_types())
    tensor_shape = (batch_size,) + dims

    model.cuda()
    #create tensor for input, centers and labels
    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda', requires_grad=True)
    float_labels = torch.zeros((batch_size,4), dtype=torch.float32, device='cuda')
    criterion = torch.nn.CrossEntropyLoss()
    for i in range(initial, iterations):
        # Get the next batch for training
        batch = eptrain.next_batch(batch_size)
        # extract labels and centers of batch datapoints
        batch.extract_labels(float_labels)
        centers = float_labels[:, 1:]
        labels = float_labels[:, 0].long().to('cuda')
        for b in range(batch_size):
            center = molgrid.float3(float(centers[b][0]), float(centers[b][1]), float(centers[b][2]))
            #intialise transformer for rotaional augmentation
            transformer = molgrid.Transform(center, 0, True)
            #center=transformer.get_quaternion().rotate(center.x,center.y,center.z)
            # random rotation on input protein
            transformer.forward(batch[b],batch[b])
            # Update input tensor with b'th datapoint of the batch
            gmaker.forward(center, batch[b].coord_sets[0], input_tensor[b])
        optimizer.zero_grad()
        # Take only the first 14 channels as that is for proteins, other 14 are ligands and will remain 0.
        output = model(input_tensor[:,:14])
        #labels_oh = nn.functional.one_hot(labels)
        #labels_oh = labels_oh
        loss = criterion(output, labels)
        loss.backward()
        predicted = torch.argmax(output,dim=1)
        #print(F.softmax(output),output, predicted, labels)
        accuracy = labels.eq(predicted).sum().float() / batch_size
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradients)
        optimizer.step()

        wandb.log({'train_loss': loss, 'train_accuracy': accuracy})
        if i % test_interval == 0 and i!=0:
            test_loss, test_accuracy,_,_,_ = test_model(model, eptest_small, gmaker,percent_reduced, batch_size)
            _, _, test_auc, test_labels, test_probs = test_model(model, eptest_large, gmaker,
                                                                                     percent_reduced, batch_size)
            scheduler.step(test_auc)
            if test_loss < Bests['test_loss']:
                Bests['test_loss'] = test_loss
                wandb.run.summary["test_loss"] = Bests['test_loss']
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'Bests': Bests,
                            'Iteration': i + 1}, outprefix + '_best_test_loss_' + str(i + 1) + '.pth.tar')
                if prev_total_loss_snap:
                    os.remove(prev_total_loss_snap)
                prev_total_loss_snap = outprefix + '_best_test_loss_' + str(i + 1) + '.pth.tar'
            if test_accuracy > Bests['test_accuracy']:
                Bests['test_accuracy'] = test_accuracy
                wandb.run.summary["test_accuracy"] = Bests['test_accuracy']
                torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'Bests': Bests,
                                'Iteration': i + 1}, outprefix + '_best_test_accuracy_' + str(i + 1) + '.pth.tar')
                if prev_total_accuracy_snap:
                    os.remove(prev_total_accuracy_snap)
                prev_total_accuracy_snap = outprefix + '_best_test_accuracy_' + str(i + 1) + '.pth.tar'
            if test_auc > Bests['test_auc']:
                Bests['test_auc'] = test_auc
                wandb.run.summary["test_auc"] = Bests['test_auc']
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'Bests': Bests,
                            'Iteration': i + 1}, outprefix + '_best_test_auc_' + str(i + 1) + '.pth.tar')
                if prev_total_auc_snap:
                    os.remove(prev_total_auc_snap)
                prev_total_auc_snap = outprefix + '_best_test_auc_' + str(i + 1) + '.pth.tar'
                Bests['train_iteration'] = i
            if i - Bests['train_iteration'] >= args.step_when and optimizer.param_groups[0]['lr'] <= ((args.step_reduce) ** args.step_end_cnt) * args.base_lr:
                last_test = 1
            print("Iteration {}, total_test_loss: {:.3f},total_test_accuracy: {:.3f},total_test_auc: {:.3f}, Best_test_loss: {:.3f},Best_test_accuracy: {:.3f},Best_test_auc: {:.3f},learning_Rate: {:.7f}".format(
                    i + 1, test_loss, test_accuracy,test_auc,Bests['test_loss'],Bests['test_accuracy'],Bests['test_auc'], optimizer.param_groups[0]['lr']))
            wandb.log({'test_loss': test_loss,'test_accuracy': test_accuracy,'test_auc':test_auc,
                       'learning rate': optimizer.param_groups[0]['lr']})
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'Bests': Bests,
                        'Iteration': i + 1}, outprefix + '_' + str(i + 1) + '.pth.tar')
            if prev_snap:
                os.remove(prev_snap)
            prev_snap = outprefix + '_' + str(i + 1) + '.pth.tar'
        if last_test:
            return Bests


if __name__ == '__main__':
    (args, cmdline) = parse_args()
    wandb.init(project="DeepPocket",name=args.run_name)
    model, gmaker, eptrain, eptest_large,eptest_small = get_model_gmaker_eproviders(args)
    Bests = train_and_test(args, model, eptrain, eptest_large,eptest_small, gmaker)
    print(Bests)
