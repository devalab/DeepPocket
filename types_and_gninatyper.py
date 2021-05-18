'''
creates types and gninatypes files of the protein for input to CNN via libmolgrid
first argument is path to protein file
second argument is path to barycenters list file
'''
import molgrid
import struct
import numpy as np
import os
import sys
def gninatype(file):
    f=open('nowat.types','w')
    f.write(file)
    f.close()
    dataloader=molgrid.ExampleProvider(shuffle=False,default_batch_size=1)
    train_types='nowat.types'
    dataloader.populate(train_types)
    example=dataloader.next()
    coords=example.coord_sets[0].coords.tonumpy()
    types=example.coord_sets[0].type_index.tonumpy()
    types=np.int_(types) 
    fout=open(file.replace('.pdb','.gninatypes'),'wb')
    for i in range(coords.shape[0]):
        fout.write(struct.pack('fffi',coords[i][0],coords[i][1],coords[i][2],types[i]))
    fout.close()
    os.remove('nowat.types')
    return file.replace('.pdb','.gninatypes')

def create_types(file,protein):
    fout=open(file.replace('.txt','.types'),'w')
    fin =open(file,'r')
    for line in fin:
        fout.write(' '.join(line.split()) + ' ' + protein +'\n')


if __name__ == '__main__':
    protein=gninatype(sys.argv[1])
    create_types(sys.argv[2],protein)
