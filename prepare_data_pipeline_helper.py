import argparse


def initialize_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, 
                        default='pdb_files/')
    parser.add_argument('--output_path', type=str, 
                        default='prepared_pdb_files/')
    parser.add_argument('--output_types_filename', type=str, 
                        default='custome_train.types')
    parser.add_argument('--output_recmolcache', type=str, 
                        default='rec.molcache2',
                        help='Filename of receptor cache')
    
    options = parser.parse_args()
    return options