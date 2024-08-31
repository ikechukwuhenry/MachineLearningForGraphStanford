# How to install Pytorch geometric on apple silicon devices
its culled from this author's guide
JG Braiser "Installing PyTorch Geometric on Mac M1 with Accelerated GPU Support"
on medium:
https://medium.com/@jgbrasier/installing-pytorch-geometric-on-mac-m1-with-accelerated-gpu-support-2e7118535c50


## Installation
1. Install homebrew:
```bash
 $ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Install miniforge3 through homebrew:
```bash
 $ brew install miniforge
 ```

3. Before using the conda command you need to source your .bashrc or .zshrc:
```bash
 $ source ~/.zshrc
```

4. Initialize conda as the base environment:
```
 $ conda init zsh
```

5. Create and activate a new conda environment:
```bash
 $ conda create --name myenv python=3.8
 $ conda activate myenv
 ```

6. Install arm64 compilers:
```
 $ conda install -y clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64
```

7. Install torch:
```
 $ MACOSX_DEPLOYMENT_TARGET=12.5 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch torchvision torchaudio
```

8. Install torch-scatter:
```
 $ MACOSX_DEPLOYMENT_TARGET=12.5 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+${cpu}.html
```

9. Install  torch-sparse:
```
 $ MACOSX_DEPLOYMENT_TARGET=12.5 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+${cpu}.html
```

9. Install  torch-cluster:
```
 $ MACOSX_DEPLOYMENT_TARGET=12.5 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-cluster -f https://data.pyg.org/whl/torch-1.12.1+${cpu}.html
```

9. Install  torch-geometric:
```
 $ MACOSX_DEPLOYMENT_TARGET=12.5 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-geometric
```

 for more detailed procedure, follow this link
 https://medium.com/@jgbrasier/installing-pytorch-geometric-on-mac-m1-with-accelerated-gpu-support-2e7118535c50

 A million thanks to JG Brasier