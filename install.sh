ENVNAME="proto-prob"

conda create --name $ENVNAME && \
eval "$(conda shell.bash hook)" && \
conda activate $ENVNAME && \

# INSTALL CONDA PACKAGES
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install lightning -c conda-forge
conda install tqdm pandas scipy -c conda-forge

# For development
conda install black isort pre-commit -c conda-forge

pre-commit install
pre-commit run

# INSTALL PIP PACKAGES
pip install ml-collections
pip install wandb
pip install -e .
