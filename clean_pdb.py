'''
Takes a PDB file and removes hetero atoms from its structure.
First argument is path to original file, second argument is path to generated file
'''
from Bio.PDB import PDBParser, PDBIO, Select
import Bio
import os
import sys

class NonHetSelect(Select):
    def accept_residue(self, residue):
        return 1 if Bio.PDB.Polypeptide.is_aa(residue,standard=True) else 0

def clean_pdb(input_file,output_file):
    pdb = PDBParser().get_structure("protein", input_file)
    io = PDBIO()
    io.set_structure(pdb)
    io.save(output_file, NonHetSelect())
    
if __name__ == '__main__':
    clean_pdb(sys.argv[1],sys.argv[2])
