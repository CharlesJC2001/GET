# GET
GET is submitted to The Visual Computer.

## Installation

The environment follows the SPFormer.

Requirements

- Python 3.x
- Pytorch 1.10
- CUDA 10.x or higher

The following installation suppose `python=3.9` `pytorch=1.10.1` and `cuda=11.3`.

- Create a conda virtual environment

  ```
  conda create -n get python=3.9
  conda activate get
  ```

- Clone the repository

  ```
  git clone https://github.com/sunjiahao1999/SPFormer.git
  ```
- Change the spformer file, the decoder file and the loss file to those in GET

- Install the dependencies

  Install [Pytorch 1.10](https://pytorch.org/)

  ```
  pip install spconv-cu113
  conda install pytorch-scatter -c pyg
  pip install -r requirements.txt
  ```

  Install segmentator from this [repo](https://github.com/Karbo123/segmentator) (We wrap the segmentator in ScanNet).

- Setup, Install spformer and pointgroup_ops.

  ```
  sudo apt-get install libsparsehash-dev
  python setup.py develop
  cd spformer/lib/
  python setup.py develop
  ```

for segmentator, cmake>=3.18, refer to [this](https://blog.csdn.net/2401_88244350/article/details/143367353), and 1.22.4<=numpy<2, refer to [this](https://blog.51cto.com/u_16175494/11068423)
, finally i chose numpy=1.24.3

for error "THC/THC.h cannot be used after pytorch-1.11", commit "#include <THC/THC.h>" in the file "./spformer/lib/pointgroup_ops/src/bfs_cluster/bfs_cluster.h"

change np.float to np.float64, change np.bool to bool, because np.float and np.bool were aborted after numpy-1.20

edit the file "./spformer/lib/pointgroup_ops/pointgroup_ops.py",change "torch.cuda.FloatTensor(size).zero_()" to "torch.zeros(size, dtype=torch.float32, device='cuda')"

## Data Preparation

### ScanNet v2 dataset

Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

Put the downloaded `scans` and `scans_test` folder as follows.

```
SPFormer
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
```

Split and preprocess data

```
cd data/scannetv2
bash prepare_data.sh
```

The script data into train/val/test folder and preprocess the data. After running the script the scannet dataset structure should look like below.

```
SPFormer
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── val_gt
```

## Ancknowledgement

Sincerely thanks for [SoftGroup](https://github.com/thangvubk/SoftGroup), [SSTNet](https://github.com/Gorilla-Lab-SCUT/SSTNet) and [SPFormer](https://github.com/sunjiahao1999/spformer) repos. This repo is build upon them.
