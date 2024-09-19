#!/bin/sh

python predict.py -p 5s1a.pdb  -c first_model_fold1_best_test_auc_85001.pth.tar -s seg0_best_test_IOU_91.pth.tar -r 3
