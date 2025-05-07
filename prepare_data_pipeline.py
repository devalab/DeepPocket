import os
from clean_pdb import clean_pdb
from create_molcache2 import create_cache2
from get_centers import get_centers
from make_types_singly import types_from_file
from prepare_data_pipeline_helper import initialize_parameters
from types_and_gninatyper import gninatype

EXTENSION_CLEAN_STEP = '_protein'

def clean_pdb_step(input_file, output_path):
    print("Starting cleaning pdb...")
    input_file_path = input_file.path
    cleaned_pdb_file_name = str(input_file.name).replace('.pdb', EXTENSION_CLEAN_STEP +".pdb")
    cleaned_pdb_file_path = os.path.join(output_path, cleaned_pdb_file_name)
    clean_pdb(input_file_path,cleaned_pdb_file_path)
    print("Finished cleaning pdb!")
    return cleaned_pdb_file_name, cleaned_pdb_file_path

def fpocket_step(cleaned_pdb_file_path):
    os.system("fpocket -f {}".format(cleaned_pdb_file_path))
    fpocket_dir = cleaned_pdb_file_path.replace('.pdb', '_out')
    fpocket_pdb_file_path = os.path.join(fpocket_dir, cleaned_pdb_file_path.split('/')[-1].split('.')[0]+'_out.pdb')
    return fpocket_dir, fpocket_pdb_file_path

def get_centers_step(fpocket_dir):
    print("Starting getting centers...")
    get_centers(fpocket_dir)
    print("Finished getting centers!")

def types_and_gninatyper_step(fpocket_pdb_file_path):
    print("Starting typing and gninatyping...")
    gninatype(fpocket_pdb_file_path)
    gninatypes_file_path = fpocket_pdb_file_path.replace('.pdb', '.gninatypes')
    print("Finished typing and gninatyping!")
    return gninatypes_file_path

def main():
    options = initialize_parameters()
    output_types_filepath = os.path.join(options.output_path, options.output_types_filename)
    output_types_file = open(output_types_filepath, 'w')

    output_recmolcache_path = os.path.join(options.output_path, options.output_recmolcache)

    gninatypes_file_path_list = []
    for input_file in os.scandir(options.input_path):
        if (input_file.is_file() == False) or (str(input_file.name).endswith('.pdb') == False):
            continue
        print(input_file.path)
        ligand_file = str(input_file.path).replace('.pdb', "_ligand.sdf")
        if os.path.isfile(ligand_file) == False:
            print("There is no ligand file for pdb file named {} with {}. Skip run".format(input_file.name, ligand_file))
            continue
        cleaned_pdb_file_name, cleaned_pdb_file_path = clean_pdb_step(input_file, options.output_path)
        fpocket_dir, fpocket_pdb_file_path = fpocket_step(cleaned_pdb_file_path)
        get_centers_step(fpocket_dir)
        bary_centers_file = os.path.join(fpocket_dir, 'bary_centers.txt')
        gninatypes_file_path = types_and_gninatyper_step(fpocket_pdb_file_path)
        gninatypes_file_path_list.append(gninatypes_file_path)
        print("Starting typing from file...")
        types_from_file(ligand_file, bary_centers_file, output_types_file)
        print("Finished typing from file!")
    output_types_file.close()
    create_cache2(gninatypes_file_path_list, '', output_recmolcache_path)


if __name__ == "__main__":
    main()