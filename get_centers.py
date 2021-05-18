'''
Takes the *_out/pockets directory from fpocket as input and outputs a file containining candidate pocket centers in that directory
'''
import os
import numpy as np
import sys
import re

def get_centers(dir):
    bary = open(dir+'/bary_centers.txt','w')
    for d in os.listdir(dir):
        centers = []
        masses = []
        if d.endswith('vert.pqr'):
            num = int(re.search(r'\d+', d).group())
            f = open(dir+'/'+d)
            for line in f:
                if line.startswith('ATOM'):
                    center=list(map(float,re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", ' '.join(line.split()[5:]))))[:3]
                    mass=float(line.split()[-1])
                    centers.append(center)
                    masses.append(mass)
            centers=np.asarray(centers)
            masses=np.asarray(masses)
            xyzm = (centers.T * masses).T
            xyzm_sum = xyzm.sum(axis=0) # find the total xyz*m for each element
            cg = xyzm_sum / masses.sum()
            bary.write(str(num) + '\t' + str(cg[0]) + '\t' + str(cg[1]) + '\t' + str(cg[2]) + '\n')
        
if __name__ == '__main__':
    get_centers(sys.argv[1]) 
