## install DROID-SLAM

We recommend recreating the environment for droidslam, which can save you a lot of unnecessary trouble.
you should install cuda-11.8 for DROID-SLAM and cuda-12.1 for DovSG, but it's not difficult

get DROID-SLAM repo

> isntall DROID-SLAM env is easy.

```bash
git submodule add https://github.com/princeton-vl/DROID-SLAM.git third_party/DROID-SLAM
git submodule update --init --recursive third_party/DROID-SLAM
cd third_party/DROID-SLAM

# there has a line need to change for depth input
# at line 90 in DovSG/third_party/DROID-SLAM/droid_slam/trajectory_filler.py
# for (tstamp, image, intrinsic) in image_stream: ==> for (tstamp, image, _, intrinsic) in image_stream:
```

```bash
conda env create -f environment.yaml
conda activate droidenv
pip install evo --upgrade --no-binary evo
pip install gdown
python setup.py install
```



