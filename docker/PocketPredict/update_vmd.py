import sys
import re

def update_vmd(bash_path, vmd_usage):
    if vmd_usage == "":
        return
    with open(bash_path, 'r') as file:
        lines = file.readlines()

    with open(bash_path, 'w') as file:
        count = 0
        for line in lines:
            if line.startswith('vmd '):
                parts = line.split(' ', 1)
                file.write(f'{vmd_usage} {parts[1]}')
            else:
                file.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_bash_file> <new_vmd_usage>")
    else:
        path = sys.argv[1]
        vmd = sys.argv[2]
        update_vmd(path, vmd)
