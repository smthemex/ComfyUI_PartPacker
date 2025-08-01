# ComfyUI_PartPacker
This is the comfyui implementation of [PartPacker](https://github.com/NVlabs/PartPacker): Efficient Part-level 3D Object Generation via Dual Volume Packing.Max varm12G


# Notice
* 新增comfyUI的标准mesh输出和适配KJ的节点的trimesh.Trimesh，方便材质赋予（注意comfyUI标准格式无法实现零件输出，只能合成为单体，最好还是用路径保存）；
* if Vram <12G make cpuoffload true ，小于12G显存开启cpu卸载
* If not connected to the mask, use the default library to obtain the mask，不连入mask则用默认库获取mask

  
# 1. Installation

In the ./ComfyUI/custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_PartPacker.git
```
---

# 2. Requirements  

```
pip install -r requirements.txt
```

# 3.Model
* 3.1.1 download  checkpoints  from [nvidia/PartPacker](https://huggingface.co/nvidia/PartPacker/tree/main) 从抱脸下载2个模型;  
* 3.1.2 dino  [facebook/dinov2-giant](https://huggingface.co/facebook/dinov2-giant/tree/main)  or [facebook/dinov2-with-registers-large](https://huggingface.co/facebook/dinov2-with-registers-large/tree/main)
```
--  ComfyUI/models/PartPacker/
    |-- flow.pth
--  ComfyUI/models/vae/
    |-- vae.pt
-- anypath/facebook/dinov2-giant/  # fix it sometimes 迟点改成单体的
   ... 
```

# Example
![](https://github.com/smthemex/ComfyUI_PartPacker/blob/main/example_workflows/example1.png)


# Acknowledgements
This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

[Dora](https://github.com/Seed3D/Dora)   
[Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)   
[Trellis](https://github.com/microsoft/TRELLIS)   

# Citation
```
@article{tang2024partpacker,
  title={Efficient Part-level 3D Object Generation via Dual Volume Packing},
  author={Tang, Jiaxiang and Lu, Ruijie and Li, Zhaoshuo and Hao, Zekun and Li, Xuan and Wei, Fangyin and Song, Shuran and Zeng, Gang and Liu, Ming-Yu and Lin, Tsung-Yi},
  journal={arXiv preprint arXiv:2506.09980},
  year={2025}
}
```
