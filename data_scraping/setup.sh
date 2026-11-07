# Install necessary packages in your virtual environment.

# Does this need to use conda?
conda install -c conda-forge grpcio
conda install -c anaconda grpcio-tools

python -m pip install -r requirements.txt

python -m pip install -e PyAnvilEditor  # For processing world folders (MC versions 1.11 and earlier only, I think)