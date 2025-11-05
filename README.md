# SAM 3D Objects

**[AI at Meta, FAIR](https://ai.meta.com/research/)**

[Xingyu Chen](https://scholar.google.com/citations?user=gjSHr6YAAAAJ&hl=en&oi=sra), [Fu-Jen Chu](https://fujenchu.github.io/), [Piotr Dollár](https://pdollar.github.io/), [Matt Feiszli](https://scholar.google.com/citations?user=A-wA73gAAAAJ&hl=en&oi=ao), [Georgia Gkioxari](https://georgiagkioxari.com/), [Pierre Gleize](https://scholar.google.com/citations?user=4imOcw4AAAAJ&hl=en&oi=ao), [Michelle Guo](https://scholar.google.com/citations?user=lyjjpNMAAAAJ&hl=en&oi=ao), [Thibaut Hardin](https://github.com/Thibaut-H), [Xiang Li](https://ryanxli.github.io/), [Kevin J Liang](https://kevinjliang.github.io/), [Aohan Lin](https://github.com/linaohan), [Jia-Wei Liu](https://jia-wei-liu.github.io/), [Ziqi Ma](https://ziqi-ma.github.io/), [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/), [Anushka Sagar](https://www.linkedin.com/in/anushkasagar/), [Alexander Sax](https://alexsax.github.io/), [Bowen Song](https://scholar.google.com/citations?user=QQKVkfcAAAAJ&hl=en&oi=sra), [Hao Tang](https://scholar.google.com/citations?user=XY6Nh9YAAAAJ&hl=en&oi=sra), [Weiyao Wang](https://sites.google.com/view/weiyaowang/home), [Xiaodong Wang](https://scholar.google.com/citations?authuser=2&user=rMpcFYgAAAAJ), [Jianing Yang](https://jedyang.com/), [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen/)

[[`<REPLACE ME Paper>`](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)] [[`<REPLACE ME Project>`](https://ai.meta.com/sam2)] [[`<REPLACE ME Demo>`](https://sam2.metademolab.com/)] [[`<REPLACE ME Dataset>`](https://ai.meta.com/datasets/segment-anything-video)] [[`<REPLACE ME Blog>`](https://ai.meta.com/blog/segment-anything-2)] [[`<REPLACE ME BibTeX>`](#citing-sam-2)]

![](doc/arch.png)

**Segment Anything 3D Models (SAM 3D)** is a pair of foundation models towards solving 3D perception from monocular views. < MORE DETAILS HERE>

## Latest updates

**10/20/2025 -- Checkpoints Launched, Web Demo and Paper are out**
- < MORE DETAILS HERE >

## Installation

< INSTALLATION INSTRUCTIONS HERE >

## Getting Started

Follow the [setup](doc/setup.md) steps before running the following.

### Single or Multi-Object 3D generation.

SAM 3D Objects can convert masked objects in an image, into 3D models with pose.

```python
# import inference code
sys.path.append("notebook")
from inference import Inference, load_image

# load model
tag = "public_v1"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/single/hf_3d_arena/a_baby_penguin_wearing_a_blue_hat.png")

# run model
output = inference(image, seed=42)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
```

Have a look at our two jupyter notebooks.
* [single object](notebook/demo_single_object.ipynb)
* [multi object](notebook/demo_multi_object.ipynb)

## Model Description

### SAM 3D checkpoints

The table below shows the SAM 3D Objects checkpoints released on October 20, 2025.

|      **Model**       | **Size (M)** |    **Speed (FPS)**     | **<Dataset> test (PCK @ 0.05)** | **<Dataset> test (MPJPE)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: |


< TODO: Update when we run speedtests >
Speed measured on an A100 with `torch 2.5.1, cuda 12.4`. See `benchmark.py` for an example on benchmarking (compiling all the model components). Compiling only the image encoder can be more flexible and also provide (a smaller) speed-up (set `compile_image_encoder: True` in the config).

## Web demo for SAM 3D

< Link to Web Demo >

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The SAM 3D project was made possible with the help of many contributors (alphabetical):

< List of collaborators here >

Third-party code: < Credit third party code here >

## Citing SAM 3D

If you use SAM 3D Objects in your research, please use the following BibTeX entry.

< TODO: Add bibtex here >
