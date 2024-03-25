conda create -c conda-forge -n deeppocket rdkit python=3.7
conda activate deeppocket
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch -y
conda install -c conda-forge fpocket -y
conda install -c conda-forge prody -y
pip install molgrid
conda install -c conda-forge biopython -y
pip install scikit-learn
pip install wandb
pip install scikit-image