import sys
import re

def update_bash_script(bash_script_path, new_protein_filename):
    pdb_pattern = re.compile(r'\S+\.pdb')

    with open(bash_script_path, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        line = pdb_pattern.sub(new_protein_filename, line)
        updated_lines.append(line)

    with open(bash_script_path, 'w') as file:
        file.writelines(updated_lines)

if __name__ == "__main__":

    bash_script_path = "start.sh"
    new_protein_filename = sys.argv[1]

    update_bash_script(bash_script_path, new_protein_filename)
