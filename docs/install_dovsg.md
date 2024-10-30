# Environment Setup

You need to get anygrasp [license and checkpoint](https://github.com/graspnet/anygrasp_sdk/blob/main/README.md#license-registration). It usually reply in 2 work days. If you do not receive the reply in 2 days, **please check the spam folder.**
You need to organize the folders like `DovSG/license`.


Then, you should install cuda-12.1 for this project
> create conda env, `python=3.9 is indeed when using anygrasp api`
```bash
conda create -n dovisg python=3.9
conda activate dovisg

git clone --recursive git@github.com:BJHYZJ/DovSG.git
cd DovSG
```



please fellow the above step to setup environment

> use cuda-12.1 and install `torch==2.3.1`
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple

```

> Install sam2
```bash
# then edit sam2's set.py
git submodule add https://github.com/facebookresearch/segment-anything-2.git third_party/segment-anything-2
cd third_party/segment-anything-2
# we use '7e1596c' branch
git checkout 7e1596c
# change setup.py
# line 27: "numpy>=1.24.4" ==> "numpy>=1.23.0",
# line 144: python_requires=">=3.10.0" ==> python_requires=">=3.9.0"

pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install -e ".[demo]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> Install groundingdino

```bash
cd ../../
git submodule add https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
cd third_party/GroundingDINO
# we use `856dde2`
git checkout 856dde2
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> Install RAM & Tag2Text:

```bash
cd ../../
git submodule add https://github.com/xinyu1205/recognize-anything.git third_party/recognize-anything
cd third_party/recognize-anything
git checkout 88c2b0c
cd ..
pip install -r ./recognize-anything/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e ./recognize-anything/ -i https://pypi.tuna.tsinghua.edu.cn/simple
```


> Install ACE

```bash
# We use ace as the relocalization model
# https://github.com/nianticlabs/ace.git
# For ease of use, we have modified some of the code, so please use the ace code provided in our repo directly
conda install opencv
cd ../ace/dsacstar
python setup.py install
```


> install `lightglue` for searching similar image 

```bash
cd ../../
git submodule add https://github.com/cvg/LightGlue.git third_party/LightGlue
cd third_party/LightGlue
git checkout edb2b83
python -m pip install -e .
```



> install Faiss
```bash
# Install the Faiss library (CPU version should be fine)
cd ../../
conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
```


> install pytorch3d

```bash
git submodule add  https://github.com/facebookresearch/pytorch3d.git third_party/pytorch3d
cd third_party/pytorch3d
git checkout 05cbea1
python setup.py install
```


> install other package

```bash
pip install ipython cmake pybind11 ninja scipy==1.10.1 scikit-learn==1.4.0 pandas==2.0.3 hydra-core opencv-python openai-clip timm matplotlib==3.7.2 imageio timm open3d numpy-quaternion more-itertools pyliblzfse einops transformers pytorch-lightning wget gdown tqdm zmq torch_geometric -i https://pypi.tuna.tsinghua.edu.cn/simple
```


> install `protobuf==3.19.0, MinkowskiEngine graspnet api`. this package is a little difficult to instll
```bash
pip install protobuf==3.19.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install git+https://github.com/pccws/MinkowskiEngine -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install graspnetAPI -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> install `torch-cluster`

```bash
# download whl file from https://pytorch-geometric.com/whl/torch-2.3.0%2Bcu121.html
pip install "/path/to/torch_cluster*.whl"
```

```bash
pip install numpy==1.23.0 supervision==0.14.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pyrealsense2 open_clip_torch graphviz alphashape pyrender -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openai==1.43.0

```

🎉 now, everything is ok, let's try it.