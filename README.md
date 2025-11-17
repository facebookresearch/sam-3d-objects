# SAM 3D Objects

**[AI at Meta, FAIR](https://ai.meta.com/research/)**

[Xingyu Chen](https://scholar.google.com/citations?user=gjSHr6YAAAAJ&hl=en&oi=sra), [Fu-Jen Chu](https://fujenchu.github.io/), [Piotr Dollár](https://pdollar.github.io/), [Matt Feiszli](https://scholar.google.com/citations?user=A-wA73gAAAAJ&hl=en&oi=ao), [Georgia Gkioxari](https://georgiagkioxari.com/), [Pierre Gleize](https://scholar.google.com/citations?user=4imOcw4AAAAJ&hl=en&oi=ao), [Michelle Guo](https://scholar.google.com/citations?user=lyjjpNMAAAAJ&hl=en&oi=ao), [Thibaut Hardin](https://github.com/Thibaut-H), [Xiang Li](https://ryanxli.github.io/), [Kevin J Liang](https://kevinjliang.github.io/), [Aohan Lin](https://github.com/linaohan), [Jia-Wei Liu](https://jia-wei-liu.github.io/), [Ziqi Ma](https://ziqi-ma.github.io/), [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/), [Anushka Sagar](https://www.linkedin.com/in/anushkasagar/), [Alexander Sax](https://alexsax.github.io/), [Bowen Song](https://scholar.google.com/citations?user=QQKVkfcAAAAJ&hl=en&oi=sra), [Hao Tang](https://scholar.google.com/citations?user=XY6Nh9YAAAAJ&hl=en&oi=sra), [Weiyao Wang](https://sites.google.com/view/weiyaowang/home), [Xiaodong Wang](https://scholar.google.com/citations?authuser=2&user=rMpcFYgAAAAJ), [Jianing Yang](https://jedyang.com/), [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen/)

[[`<REPLACE ME Paper>`](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)] [[`<REPLACE ME Project>`](https://ai.meta.com/sam2)] [[`<REPLACE ME Demo>`](https://sam2.metademolab.com/)] [[`<REPLACE ME Dataset>`](https://ai.meta.com/datasets/segment-anything-video)] [[`<REPLACE ME Blog>`](https://ai.meta.com/blog/segment-anything-2)] [[`<REPLACE ME BibTeX>`](#citing-sam-2)]

![](doc/arch.png)

**SAM 3D Objects** is a foundation model that reconstructs full 3D shape geometry, texture, and layout from a single image, excelling in real-world scenarios with occlusion and clutter by using progressive training and a data engine with human feedback. It outperforms prior 3D generation models in human preference tests on real-world objects and scenes. We released code, weights, online demo, and a new challenging benchmark.

## Latest updates

**11/19/2025** - Checkpoints Launched, Web Demo and Paper are out.

## Getting Started

Follow the [setup](doc/setup.md) steps before running the following.

### Single or Multi-Object 3D generation.

SAM 3D Objects can convert masked objects in an image, into 3D models with pose.

```python
# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# run model
output = inference(image, mask, seed=42)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
```

Have a look at our two jupyter notebooks.
* [single object](notebook/demo_single_object.ipynb)
* [multi object](notebook/demo_multi_object.ipynb)

## SAM 3D Body

[SAM 3D Body (3DB)](https://github.com/facebookresearch/sam-3d-body) is a robust promptable foundation model for single-image 3D human mesh recovery (HMR).

As a way to combine the strengths of both **SAM 3D Objects** and **SAM 3D Body**, we provide an example notebook that demonstrates how to combine the results of both models such that they are aligned in the same frame of reference. Check it out [here](notebook/demo_3db_mesh_alignment.ipynb).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citing SAM 3D Objects

If you use SAM 3D Objects in your research, please use the following BibTeX entry.

< TODO: Add bibtex here >
