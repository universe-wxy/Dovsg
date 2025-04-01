## install DROID-SLAM

We recommend recreating the environment for droidslam, which can save you a lot of unnecessary trouble.
you should install cuda-11.8 for DROID-SLAM and cuda-12.1 for DovSG, but it's not difficult

***Important***: Please change `CUDA_HOME` to `cuda-11.8`, Then
```
source ~/.bashrc
```

> Change `rgb` Mode to `rgbd` Mode.

```bash
# we use branch ``8016d2b
cd third_party/DROID-SLAM

# (NOTE) there has a line need to change for depth input
# at line 90 in DROID-SLAM/droid_slam/trajectory_filler.py
# for (tstamp, image, intrinsic) in image_stream: ==> for (tstamp, image, _, intrinsic) in image_stream:
```

> Deactivate DovSG Environment
```bash
conda deactivate dovsg
```

> Install DROID-SLAM Conda Environment

```bash
conda create -n droidenv python=3.9 -y
conda activate droidenv
```

> Install pytorch=1.10 torchvision torchaudio cudatoolkit
```bash
conda install pytorch=1.10 torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
```

> Install the remaining libraries

```bash
conda install suitesparse -c conda-forge -y
pip install open3d==0.15.2 scipy opencv-python==4.7.0.72 matplotlib pyyaml==6.0.2 tensorboard # -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install evo --upgrade --no-binary evo
pip install gdown
pip install numpy==1.23.0 numpy-quaternion==2023.0.4
```

> Install torch-sactter
You can download `torch-sactter` from [here](https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl)

```bash
wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
```


> Install DROID-SLAM
```bash
# You should install gcc-10/g++10
sudo apt install gcc-10 g++-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
```
```bash
python setup.py install
```




Then, please download the checkpoints by following the instructions in [download_checkpoints.md](down_checkpoints.md)