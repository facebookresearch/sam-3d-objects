# SAM 3D Objects (Multi-View Extension)

This is a fork of [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) with **multi-view 3D reconstruction** support.

## 🆕 What's New

**Multi-View 3D Reconstruction**: This fork adds training-free multi-view inference capability using a multidiffusion approach. You can now generate consistent 3D models from multiple input images of the same object from different viewpoints, without requiring model retraining.

### Results Comparison

The following comparison demonstrates the improvement of multi-view reconstruction over single-view reconstruction:

<table>
<tr>
  <td align="center" width="33%"><b>Single-View (View 3)</b></td>
  <td align="center" width="33%"><b>Single-View (View 6)</b></td>
  <td align="center" width="33%"><b>Multi-View (All 8 Views)</b></td>
</tr>
<tr>
  <td align="center" width="33%" style="padding: 5px;">
    <b>Input Image</b><br>
    <img src="data/example/images/3.png" width="100%" style="max-width: 300px;"/>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <b>Input Image</b><br>
    <img src="data/example/images/6.png" width="100%" style="max-width: 300px;"/>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <b>Input Images</b><br>
    <table width="100%" cellpadding="2" cellspacing="2">
    <tr>
      <td align="center"><img src="data/example/images/1.png" width="60px"/></td>
      <td align="center"><img src="data/example/images/2.png" width="60px"/></td>
      <td align="center"><img src="data/example/images/3.png" width="60px"/></td>
      <td align="center"><img src="data/example/images/4.png" width="60px"/></td>
    </tr>
    <tr>
      <td align="center"><img src="data/example/images/5.png" width="60px"/></td>
      <td align="center"><img src="data/example/images/6.png" width="60px"/></td>
      <td align="center"><img src="data/example/images/7.png" width="60px"/></td>
      <td align="center"><img src="data/example/images/8.png" width="60px"/></td>
    </tr>
    </table>
  </td>
</tr>
<tr>
  <td align="center" colspan="3">
    <b>↓ 3D Reconstruction ↓</b>
  </td>
</tr>
<tr>
  <td align="center" width="33%" style="padding: 5px;">
    <b>3D Result</b><br>
    <video width="100%" controls style="max-width: 300px;">
      <source src="data/example/visualization_results/view3_cropped.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <b>3D Result</b><br>
    <video width="100%" controls style="max-width: 300px;">
      <source src="data/example/visualization_results/view6_cropped.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <b>3D Result</b><br>
    <video width="100%" controls style="max-width: 300px;">
      <source src="data/example/visualization_results/all_views_cropped.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </td>
</tr>
<tr>
  <td align="left" width="33%" style="padding: 10px;">
    <small><b>Analysis:</b> Due to occlusion in the input image, the red collar on the dog is not visible, resulting in its absence in the generated 3D model.</small>
  </td>
  <td align="left" width="33%" style="padding: 10px;">
    <small><b>Analysis:</b> Many frontal parts of the dog are occluded or not visible from this angle, leading to structural errors in the front-facing regions of the generated model.</small>
  </td>
  <td align="left" width="33%" style="padding: 10px;">
    <small><b>Analysis:</b> By combining information from all 8 views, the multi-view reconstruction produces a complete and accurate 3D model that closely matches the actual object.</small>
  </td>
</tr>
</table>

### Quick Start

```bash
# Multi-view reconstruction (mask_prompt=None, images and masks in same directory)
python run_inference.py --input_path ./data/images_and_masks

# Single-view reconstruction (specify a single image name)
python run_inference.py --input_path ./data/images_and_masks --image_names image1

# Multi-view reconstruction (mask_prompt!=None, images in images/, masks in {mask_prompt}/)
python run_inference.py --input_path ./data --mask_prompt stuffed_toy

# Specify multiple image names (can be any filename without extension)
python run_inference.py --input_path ./data --mask_prompt stuffed_toy --image_names image1,view_a,2
```

See the [Multi-View 3D Reconstruction](#multi-view-3d-reconstruction) section below for detailed documentation.

---

# SAM 3D

SAM 3D Objects is one part of SAM 3D, a pair of models for object and human mesh reconstruction.  If you're looking for SAM 3D Body, [click here](https://github.com/facebookresearch/sam-3d-body).

# SAM 3D Objects

**SAM 3D Team**, [Xingyu Chen](https://scholar.google.com/citations?user=gjSHr6YAAAAJ&hl=en&oi=sra)\*, [Fu-Jen Chu](https://fujenchu.github.io/)\*, [Pierre Gleize](https://scholar.google.com/citations?user=4imOcw4AAAAJ&hl=en&oi=ao)\*, [Kevin J Liang](https://kevinjliang.github.io/)\*, [Alexander Sax](https://alexsax.github.io/)\*, [Hao Tang](https://scholar.google.com/citations?user=XY6Nh9YAAAAJ&hl=en&oi=sra)\*, [Weiyao Wang](https://sites.google.com/view/weiyaowang/home)\*, [Michelle Guo](https://scholar.google.com/citations?user=lyjjpNMAAAAJ&hl=en&oi=ao), [Thibaut Hardin](https://github.com/Thibaut-H), [Xiang Li](https://ryanxli.github.io/)⚬, [Aohan Lin](https://github.com/linaohan), [Jia-Wei Liu](https://jia-wei-liu.github.io/), [Ziqi Ma](https://ziqi-ma.github.io/)⚬, [Anushka Sagar](https://www.linkedin.com/in/anushkasagar/), [Bowen Song](https://scholar.google.com/citations?user=QQKVkfcAAAAJ&hl=en&oi=sra)⚬, [Xiaodong Wang](https://scholar.google.com/citations?authuser=2&user=rMpcFYgAAAAJ), [Jianing Yang](https://jedyang.com/)⚬, [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen/)⚬, [Piotr Dollár](https://pdollar.github.io/)†, [Georgia Gkioxari](https://georgiagkioxari.com/)†, [Matt Feiszli](https://scholar.google.com/citations?user=A-wA73gAAAAJ&hl=en&oi=ao)†§, [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)†§

***Meta Superintelligence Labs***

*Core contributor (Alphabetical, Equal Contribution), ⚬Intern, †Project leads, §Equal Contribution

[[`Paper`](https://ai.meta.com/research/publications/sam-3d-3dfy-anything-in-images/)] [[`Code`](https://github.com/facebookresearch/sam-3d-objects)] [[`Website`](https://ai.meta.com/sam3d/)] [[`Demo`](https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d)] [[`Blog`](https://ai.meta.com/blog/sam-3d/)] [[`BibTeX`](#citing-sam-3d-objects)]

**SAM 3D Objects** is a foundation model that reconstructs full 3D shape geometry, texture, and layout from a single image, excelling in real-world scenarios with occlusion and clutter by using progressive training and a data engine with human feedback. It outperforms prior 3D generation models in human preference tests on real-world objects and scenes. We released code, weights, online demo, and a new challenging benchmark.


<p align="center"><img src="doc/intro.png"/></p>

-----

<p align="center"><img src="doc/arch.png"/></p>

## Latest updates

**11/19/2025** - Checkpoints Launched, Web Demo and Paper are out.

## Installation

Follow the [setup](doc/setup.md) steps before running the following.

## Single or Multi-Object 3D Generation

SAM 3D Objects can convert masked objects in an image, into 3D models with pose, shape, texture, and layout. SAM 3D is designed to be robust in challenging natural images, handling small objects and occlusions, unusual poses, and difficult situations encountered in uncurated natural scenes like this kidsroom:

<p align="center">
  <img src="notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png" width="55%"/>
  <img src="doc/kidsroom_transparent.gif" width="40%"/>
</p>

For a quick start, run `python demo.py` or use the the following lines of code:

```python
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image and mask
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# run model
output = inference(image, mask, seed=42)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
```

For  more details and multi-object reconstruction, please take a look at out two jupyter notebooks:
* [single object](notebook/demo_single_object.ipynb)
* [multi object](notebook/demo_multi_object.ipynb)

## Multi-View 3D Reconstruction

SAM 3D Objects now supports multi-view 3D reconstruction using a training-free multidiffusion approach. This allows you to generate consistent 3D models from multiple input images of the same object from different viewpoints.

### Quick Start

Use the `run_inference.py` script for both single-view and multi-view reconstruction:

```bash
# Multi-view reconstruction (mask_prompt=None, images and masks in same directory)
python run_inference.py --input_path ./data/images_and_masks

# Single-view reconstruction (specify a single image name)
python run_inference.py --input_path ./data/images_and_masks --image_names image1

# Multi-view reconstruction (mask_prompt!=None, images in images/, masks in {mask_prompt}/)
python run_inference.py --input_path ./data --mask_prompt stuffed_toy

# Specify multiple image names (can be any filename without extension)
python run_inference.py --input_path ./data --mask_prompt stuffed_toy --image_names image1,view_a,2
```

### Data Structure

Multi-view data can be organized in two ways:

**Structure 1** (when `mask_prompt=None`): Images and masks in the same directory
```
input_path/
    ├── 1.png          # Original image (PNG format)
    ├── 1_mask.png     # Mask (RGBA format, alpha channel stores mask info)
    ├── 2.png
    ├── 2_mask.png
    └── ...
```

**Structure 2** (when `mask_prompt!=None`, e.g., `mask_prompt="stuffed_toy"`): Images and masks in separate directories
```
input_path/
    ├── images/
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    └── stuffed_toy/  (or {mask_prompt}/)
        ├── 1.png (or 1_mask.png)
        ├── 2.png (or 2_mask.png)
        └── ...
```

**Mask Format**: RGBA format where the alpha channel stores mask information (alpha=255 for object, alpha=0 for background).

### Command Line Options

Run `python run_inference.py --help` for full documentation. Key parameters:

- `--input_path`: Path to input directory (required)
- `--mask_prompt`: Mask folder name. If None, images and masks are in the same directory; if specified, images are in `input_path/images/` and masks are in `input_path/{mask_prompt}/`
- `--image_names`: Image names (without extension), e.g., `"image1,view_a"` or `"1,2"` or `"image1"`. Can specify multiple, comma-separated. If not specified, uses all available images
- `--decode_formats`: Output formats, e.g., `"gaussian,mesh"` or `"gaussian"` (default: `gaussian,mesh`)
- `--seed`: Random seed (default: 42)
- `--stage1_steps`: Stage 1 inference steps (default: 50)
- `--stage2_steps`: Stage 2 inference steps (default: 25)
- `--model_tag`: Model tag (default: hf)

The script automatically detects whether to use single-view or multi-view inference based on the number of views provided. Multi-view reconstruction uses a training-free multidiffusion approach to fuse predictions from all views.

## SAM 3D Body

[SAM 3D Body (3DB)](https://github.com/facebookresearch/sam-3d-body) is a robust promptable foundation model for single-image 3D human mesh recovery (HMR).

As a way to combine the strengths of both **SAM 3D Objects** and **SAM 3D Body**, we provide an example notebook that demonstrates how to combine the results of both models such that they are aligned in the same frame of reference. Check it out [here](notebook/demo_3db_mesh_alignment.ipynb).

## License

The SAM 3D Objects model checkpoints and code are licensed under [SAM License](./LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The SAM 3D Objects project was made possible with the help of many contributors.

Robbie Adkins,
Paris Baptiste,
Karen Bergan,
Kai Brown,
Michelle Chan,
Ida Cheng,
Khadijat Durojaiye,
Patrick Edwards,
Daniella Factor,
Facundo Figueroa,
Rene  de la Fuente,
Eva Galper,
Cem Gokmen,
Alex He,
Enmanuel Hernandez,
Dex Honsa,
Leonna Jones,
Arpit Kalla,
Kris Kitani,
Helen Klein,
Kei Koyama,
Robert Kuo,
Vivian Lee,
Alex Lende,
Jonny Li,
Kehan Lyu,
Faye Ma,
Mallika Malhotra,
Sasha Mitts,
William Ngan,
George Orlin,
Peter Park,
Don Pinkus,
Roman Radle,
Nikhila Ravi,
Azita Shokrpour,
Jasmine Shone,
Zayida Suber,
Phillip Thomas,
Tatum Turner,
Joseph Walker,
Meng Wang,
Claudette Ward,
Andrew Westbury,
Lea Wilken,
Nan Yang,
Yael Yungster


## Citing SAM 3D Objects

If you use SAM 3D Objects in your research, please use the following BibTeX entry.

< TODO: Add bibtex here >
