import os
import sys
import numpy as np
import time
import rdkit
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import AllChem


def types_from_file(ligand_file, bary_centers_file, output_types_file):
    distance = 4
    atom_nps=[]
    mol = Chem.MolFromMolFile(ligand_file,sanitize=False)
    c=mol.GetConformer()
    atom_np=c.GetPositions()
    atom_nps.append(atom_np)
    centers = np.loadtxt(bary_centers_file)
    if centers.shape[0]==4 and len(centers.shape)==1:
        centers=np.expand_dims(centers,axis=0)
    limit = centers.shape[0]
    if not (centers.shape[0]==0 and len(centers.shape)==1):
        sorted_centers = centers[:int(limit), 1:]
        for i in range(int(limit)):
            label=0
            for atom_np in atom_nps:
                dist = np.linalg.norm((atom_np - sorted_centers[i,:]), axis=1)
                rel_centers = np.where(dist <= float(distance), 1, 0)
                if (np.count_nonzero(rel_centers) > 0):
                    label =1
                else:
                    label =0
                output_types_file.write(str(label)+ ' '+ str(sorted_centers[i][0])+ ' '+ str(sorted_centers[i][1])+ ' '+ str(sorted_centers[i][2])+ ' '+prot+'/'+prot+'_protein_nowat.gninatypes\n')

def main():
    output_types_file = open('pdbbind_train.types','w')

if __name__ == "__main__":
    main()