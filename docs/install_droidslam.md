## install DROID-SLAM

We recommend recreating the environment for droidslam, which can save you a lot of unnecessary trouble.
you should install cuda-11.8 for DROID-SLAM and cuda-12.1 for DovSG, but it's not difficult

get DROID-SLAM repo

> isntall DROID-SLAM env is easy.

```bash
# we use branch ``8016d2b
cd third_party/DROID-SLAM

# there has a line need to change for depth input
# at line 90 in DROID-SLAM/droid_slam/trajectory_filler.py
# for (tstamp, image, intrinsic) in image_stream: ==> for (tstamp, image, _, intrinsic) in image_stream:
```

```bash
# DROID-SLAM need cuda low than 12.1, we use cuda-11.8, so you may need to edit ~/.bashrc
conda env create -f environment.yaml
conda activate droidenv
pip install evo --upgrade --no-binary evo
pip install gdown
python setup.py install
```

🎉 now, everything is ok, let's try it.

