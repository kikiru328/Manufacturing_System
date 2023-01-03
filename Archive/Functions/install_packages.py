import os

os.system('conda create -n manufacturing python=3.8 -y')
os.system('conda activate manufacturing')
os.system('conda install -c conda-forge lap -y')
os.system('pip install cython-bbox-windows')
os.system(f"pip install -r {os.path.realpath('../requirements.txt')}")