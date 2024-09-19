import subprocess
import os

def singular(protein_name, dir_name, vmd_usage):
    if not protein_name:
        print("Usage: Enter the name of the protein.")
        return
    
    protein_file = protein_name
    protein_name = os.path.splitext(protein_file)[0]

    container_name = ""
    
    subprocess.run(['python', 'update_protein.py', f'{protein_file}'], check=True)
    subprocess.run(['docker', 'build', '-t', 'front', '.'], check=True)

    try:
        subprocess.run(['docker', 'volume', 'create', 'pock'], check=True)
        container_name = protein_file.replace('.pdb', '_cont')
        subprocess.run(['docker', 'run', '--name', f'{container_name}', '-it', '-v', 'pock:/front', 'front'], check=True)
        subprocess.run(['docker', 'cp', f'{container_name}:/front/{protein_name}_nowat_out', '.'], check=True)
        subprocess.run(['docker', 'rm', '-f', container_name], check=True)
        subprocess.run(['python', 'update_vmd.py', f'{protein_name}_nowat_out/{protein_name}_nowat_VMD.sh', f'{vmd_usage}'])
        os.chdir(f'{protein_name}_nowat_out')
        subprocess.run([f'./{protein_name}_nowat_VMD.sh'], check=True)
        os.chdir('..')
        subprocess.run(['docker', 'volume', 'remove', 'pock'], check=True)

        if dir_name != "":
            subprocess.run(['cp', '-r', f'{protein_name}_nowat_out', f'{dir_name}'])
            subprocess.run(['rm', '-rf', f'{protein_name}_nowat_out'])
            subprocess.run(['rm', '-rf', f'{protein_name}.pdb'])

    except:
        subprocess.run(['docker', 'rm', f'{container_name}'], check=True)
        subprocess.run(['docker', 'volume', 'remove', 'pock'], check=True)

def entire(protein_name, dir_name):
    if not protein_name:
        print("Usage: Enter the name of the protein.")
        return
    
    protein_file = protein_name
    protein_name = os.path.splitext(protein_file)[0]

    container_name = ""
    
    subprocess.run(['python', 'update_protein.py', f'{protein_file}'], check=True)
    subprocess.run(['docker', 'build', '-t', 'front', '.'], check=True)

    try:
        subprocess.run(['docker', 'volume', 'create', 'pock'], check=True)
        container_name = protein_file.replace('.pdb', '_cont')
        subprocess.run(['docker', 'run', '--name', f'{container_name}', '-it', '-v', 'pock:/front', 'front'], check=True)
        subprocess.run(['docker', 'cp', f'{container_name}:/front/{protein_name}_nowat_out', '.'], check=True)
        subprocess.run(['docker', 'rm', '-f', container_name], check=True)
        subprocess.run(['docker', 'volume', 'remove', 'pock'], check=True)

        if dir_name != "":
            subprocess.run(['cp', '-r', f'{protein_name}_nowat_out', f'{dir_name}'])
            subprocess.run(['rm', '-rf', f'{protein_name}_nowat_out'])
            subprocess.run(['rm', '-rf', f'{protein_name}.pdb'])

    except:
        subprocess.run(['docker', 'rm', f'{container_name}'], check=True)
        subprocess.run(['docker', 'volume', 'remove', 'pock'], check=True)

    

if __name__ == "__main__":

    vmd = ""
    two = 0

    while (True):
        print("\n")
        print("-----------------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------------")
        print("""Choose the option that suits your requirements:
              
          1) Set the path to the VMD executable, in order to run VMD (default command is simply `vmd`).
          2) Set the destination of generated files (default storage location is the directory containing the protein file).
          3) Enter the path of a singular pdb file to run the prediction model on.
          4) Run the prediction model on an entire directory of proteins.
          5) Exit the program.\n""")
    
        choice = int(input("Enter the serial corresponding to your choice: "))
        print()

        if choice == 1:
            vmd = input("Enter the path of the vmd executable, to run VMD from a shell script: ")

        if choice == 2:
            dir_name = input("Enter the directory where you want to store the generated files: ")
            two = 1

        if choice == 3:
            path = input("Enter the .pdb file you want to run the prediction model on: ")
            subprocess.run(['cp', f'{path}', '.'])
            protein_name = os.path.basename(path)
            if two == 0:
                dir_name = os.path.dirname(path)
            print(dir_name)
            singular(protein_name, dir_name, vmd)

        if choice == 4:
            directory = input("Enter the directory you want to use proteins from: ")
            if not os.path.isdir(directory):
                print(f"\nThe directory {directory} does not exist.")
                continue
            
            for file_name in os.listdir(directory):
                if file_name.endswith('.pdb'):
                    path = os.path.join(directory, file_name)
                    subprocess.run(['cp', f'{path}', '.'])
                    protein_name = os.path.basename(path)
                    if two == 0:
                        dir_name = os.path.dirname(path)
                    entire(protein_name, dir_name)

        if choice == 5:
            break

