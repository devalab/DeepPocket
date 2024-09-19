# Dockerized DeepPocket

DeepPocket is a 3D convolutional neural network framework for ligand binding site detection and segmentation from protein structures. This directory offers a Dockerized implementation of DeepPocket’s pocket prediction and ranking features, providing a consistent, hassle-free user experience by containerizing all dependencies.

## Features

- **Ligand Binding Site Detection:** Identify potential binding sites in protein structures.
- **Segmentation:** Perform detailed segmentation of the identified binding sites.
- **Dockerized Environment:** Run the application within a Docker container to avoid dependency or system issues.
- **Interactive Menu:** A user-friendly interface for running predictions on individual proteins or entire directories.
- **Visualization:** Automatically view classified pockets using Visual Molecular Dynamics (VMD).

## Prerequisites

- **Docker:** Ensure Docker is installed. You can find installation instructions [here](https://docs.docker.com/get-docker/).

- **Downloading models:** Download the classification and segmentation models from [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/rishal_aggarwal_alumni_iiit_ac_in/EoJSrvuiKPlAluOJLjTzfpcBT2fVRdq8Sr4BMmil0_tvHw?e=kXUss4). From `classification_models` folder, download the `first_model_fold1_best_test_auc_85001.pth.tar` file. From `segmentation_models` folder, download the `seg0_best_test_IOU_91.pth.tar` file. Place these files in the `PocketPredict` directory.

## Getting Started

### Running the Application

1. **Make the script executable:**

   ```bash
   chmod +x run.sh
   ```

2. **Execute the script:**

   ```bash
   ./run.sh
   ```

### Using the Application

The menu-driven interface will guide you through the process of running predictions on individual proteins or entire directories. Follow the on-screen instructions to select the desired options:

- **Option 1:** Set the path to the VMD executable for visualization. Provide the absolute path (e.g., `/home/my-system/vmd-1.9.4a57/bin/vmd`). If not specified, it defaults to the `vmd` command.
  
- **Option 2:** Specify a destination directory for storing results. Provide the absolute path (e.g., `/home/my-system/results`). If not specified, results are stored in the same directory as the input file.

- **Option 3:** Run predictions on individual proteins. You will be prompted to enter the path to the protein structure file (PDB format) (e.g., `/home/my-system/proteins/1abc.pdb`). Results are stored in the specified directory, and visualization will be displayed using VMD.

- **Option 4:** Run predictions on an entire directory of proteins. Enter the path to the directory containing PDB files (e.g., `/home/my-system/proteins`). Results are stored in the specified directory, but visualization is not displayed automatically. You can visualize the results later using the generated VMD files.

## Notes on Functionality

- When running the model on a single protein, the visualization of the protein, including the classified and segmented pockets, automatically pops up in a VMD window.
- For batch processing (entire directories), the results can be visualized manually later using the generated VMD visualization scripts.
- In order to change the default models that are being used, you can go to `PocketPredict/start.sh` and change the model paths accordingly.
