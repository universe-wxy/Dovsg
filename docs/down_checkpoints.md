In our project, a total of 7 models are used. The versions and download links/methods for each model are as follows:
1. anygrasp: when you get anygrasp license from <a herf="https://github.com/graspnet/anygrasp_sdk/blob/main/README.md#license-registration">here</a>, it will provid checkpoint for you.
2. bert-base-uncased: [https://huggingface.co/google-bert/bert-base-uncased
3. CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/google-bert/bert-base-uncased)
4. droid-slam: [https://drive.google.com/file/u/0/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing&pli=1](https://drive.google.com/file/u/0/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing&pli=1)
5. GroundingDINO: [https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) and [https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinT_OGC.py](https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinT_OGC.py)
6. recognize_anything: [https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/ram_swin_large_14m.pth](https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/ram_swin_large_14m.pth)
7. segment-anything-2: [https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints](https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints)

<!-- Alternatively, you can download all the checkpoints we use in the project from <a herf="">here</a>. Note that for the anygrasp model, you will need to obtain a custom license and checkpoint based on your device ID. -->

You should organize the checkpoints as follows:
```bash
DovSG/
    ├──checkpoints
    │    ├──anygrasp
    │    ├──bert-base-uncased
    │    ├──CLIP-ViT-H-14-laion2B-s32B-b79K
    │    ├──droid-slam
    │    ├──GroundingDINO
    │    ├──recognize_anything
    │    └──segment-anything-2
    ├──license  # anygrasp license
    ...  
```