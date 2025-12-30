<h1 align="center">MEGA-PCC: A Mamba-based Efficient Approach for Joint Geometry and Attribute Point Cloud Compression </h1>
<p align="center">
    <strong><a href="https://github.com/kai0416s">Kai-Hsiang Hsieh</a></strong><sup>1</sup>,
    <strong><a href="mailto:yimmonyneath@gmail.com">Monyneath Yim</a></strong><sup>1</sup>, 
    <strong><a href="https://sites.google.com/g2.nctu.edu.tw/wpeng">Wen-Hsiao Peng</a></strong><sup>2</sup>,
    <strong><a href="https://chiang.ccu.edu.tw/index.php">Jui-Chiu Chiang</a></strong><sup>1</sup>,
</p>

<p align="center">
    <sup>1</sup>National Chung Cheng University, Taiwan<br>
    <sup>2</sup>National Yang Ming Chiao Tung University, Taiwan
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2512.22463"><img src="https://img.shields.io/badge/Arxiv-2512.22463-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
  <a href="https://github.com/kai0416s/MEGA-PCC/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"></a>
  <a href="https://kai0416s.github.io/MEGA-PCC/"><img src="https://img.shields.io/badge/MEGA-PCC?label=Project%20Page" alt="Home Page"></a>
</p>

# [WACV 2026] Official Implementation for MEGA-PCC

* MEGA-PCC introduces a novel single-encoder, dualdecoder design that jointly compresses geometry and attributes, avoiding both the recoloring step and the need
for coordinating separate models.
* MEGA-PCC leverages Mamba architecture to enhance both the encoder/decoder and the entropy model, improving the modeling of long-range dependencies and boosting compression performance.
* MEGA-PCC supports fully end-to-end training and learns to allocate bits between geometry and attributes without requiring exhaustive model matching, thus achieving efficient and fast compression.

## Todo
- [x] ~~Release the Paper~~
- [ ] Release inference code
- [ ] Release checkpoint
- [ ] Release training code

# Abstract
Joint compression of point cloud geometry and attributes is essential for efficient 3D data representation. Existing methods often rely on post-hoc recoloring procedures and manually tuned bitrate allocation between geometry and attribute bitstreams in inference, which hinders end-to-end optimization and increases system complexity. 

To overcome these limitations, we propose **MEGA-PCC**, a fully end-to-end, learning-based framework featuring two specialized models for joint compression. The main compression model employs a shared encoder that encodes both geometry and attribute information into a unified latent representation, followed by dual decoders that sequentially reconstruct geometry and then attributes. 

Complementing this, the **Mamba-based Entropy Model (MEM)** enhances entropy coding by capturing spatial and channel-wise correlations to improve probability estimation. Both models are built on the Mamba architecture to effectively model long-range dependencies and rich contextual features. 

By eliminating the need for recoloring and heuristic bitrate tuning, MEGA-PCC enables data-driven bitrate allocation during training and simplifies the overall pipeline. 
Extensive experiments demonstrate that MEGA-PCC achieves superior rate-distortion performance and runtime efficiency compared to both traditional and learning-based baselines, offering a powerful solution for AI-driven point cloud compression.


![architecture](https://github.com/kai0416s/MEGA-PCC/blob/main/static/images/MEGA_Architecture.png)

# Result
To comprehensively assess the effectiveness of our proposed MEGA-PCC method, we compare it against the classical standards G-PCCv23 and V-PCCv22, as well as recent learning-based joint compression methods, including YOGA, DeepPCC, Unicorn, and JPEG Pleno VM4.1.

![BDBR](https://github.com/kai0416s/MEGA-PCC/blob/main/static/images/MEGA_BDBR.png)


![RD-Curve](https://github.com/kai0416s/MEGA-PCC/blob/main/static/images/RD.png)

# Complexity Analysis
MEGAPCC offers an excellent trade-off between model size, runtime, and performance. It achieves the smallest model size
(44.0 MB) and the lowest encoding and decoding times among GPU-based methods, while outperforming YOGA and DeepPCC in terms of 1-PCQM performance. 

Note that the reported encoding/decoding times for these learned PCC methods exclude recoloring time. MEGA-PCC avoids this processing entirely—requiring no recoloring or bit allocation—thanks to its unified design. The significant gap in runtime, faster decoding compared to existing methods clearly highlights MEGA-PCC’s practicality for real-time applications.

![RD-Curve](https://github.com/kai0416s/MEGA-PCC/blob/main/static/images/complex_MEGA.png)

# ⚙️Requirments environment
* Create env：
```
conda create -n MEGAPCC python=3.8
conda activate MEGAPCC
conda install MEGAPCC devel -c anaconda
```
- Install torch：
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```
* ## install MinkowskiEngine：
[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
* ## Requirements
Step 1. Install requirements:
```
pip install -r requirements.txt
```
Step 2. Use the torchacc
```
cd torchacc/
python3 setup.py install
cd ../
```

# dataset
Training：

* [ScanNet dataset](https://github.com/ScanNet/ScanNet), which is a large open-source dataset of indoor scenes.

* [RWTT dataset](https://texturedmesh.isti.cnr.it/), consists of 568 textured models generated using various 3D reconstruction methods, known for their detailed color and texture information.

>cube division with size 128* 128 *128. We randomly selected 20,000 cubes and used them for training.

![trainingdata](https://github.com/kai0416s/MEGA-PCC/blob/main/static/images/traindata.png)

- Testing：
8iVFB dataset(longdress, loot, redandblack, soldier, basketball_player, and dancer.)

![testingdata](https://github.com/kai0416s/MEGA-PCC/blob/main/static/images/testdata.png)

# check point download(coming soon)：
| check point  | [Link]()|
| ---------- | -----------|


## ❗ After data preparation, the overall directory structure should be：
```
│MEGA-PCC/
├──results/
├──output/
├──ckpts/
│   ├──/final_result/
│                 ├──/R6.pth
├──.......
```

# Training
* The default setting：

Epoch: 60

The learning rate is initialized at 4e-5 and halved every 20 epochs until it decreases to 5e-5.

| Parameter | R6 | R5 | R4 | R3 | R2 | R1 |
|----------------------|----|----|----|----|----|----|
| lamda_A               | 0.03 | 0.0195 | 0.013975 | 0.0085 | 0.0032 | 0.00135 |
| lamda_G               | 7.2    | 1.625    | 0.7525    | 0.75    |  0.36   | 0.225   |

## Train
```
python train.py
```
- You need to change the check point location and then can train the low rate check point.
```
parser.add_argument("--init_ckpt", default='/MEGAPCC/ckpts/final_result/R6.pth')
```
# Testing

* input the orignal point cloud path：
```
filedir_list = [
  './testdata/8iVFB/longdress_vox10_1300.ply',
  './testdata/8iVFB/loot_vox10_1200.ply',
  './testdata/8iVFB/redandblack_vox10_1550.ply',
  './testdata/8iVFB/soldier_vox10_0690.ply',
  './testdata/Owlii/basketball_player_vox11_00000200.ply',
  './testdata/Owlii/dancer_vox11_00000001.ply',
]
```
- output path and check point location：
```
Output = '/1230'
Ckpt = '/final_result'
```
* The check point we have provide：
```
ckptdir_list = [
  './ckpts' + Ckpt + '/R1.pth',
  './ckpts' + Ckpt + '/R2.pth',
  './ckpts' + Ckpt + '/R3.pth',
  './ckpts' + Ckpt + '/R4.pth',
  './ckpts' + Ckpt + '/R5.pth',
  './ckpts' + Ckpt + '/R6.pth',
]
```
> R6.pth is the high rate.

Then you can run the test and get the result in folder 1226 and we also provide the experiment result.

## Test
```
python test.py
```

# Link
We welcome readers to refer to our previous work on point cloud compression.
* [ANF-PCAC](https://github.com/Applied-Computing-and-Multimedia-Lab/ANF-PCAC)： Sparse Tensor-based point cloud attribute compression using Augmented Normalizing Flows
* [SEDD-PCC](https://github.com/Applied-Computing-and-Multimedia-Lab/SEDD-PCC)： A Single Encoder-Dual Decoder Framework For End-To-End Learned Point Cloud Compression

# Authors
These files are provided by National Chung Cheng University [Applied Computing and Multimedia Lab](https://chiang.ccu.edu.tw/index.php).

Please contact us (s69924246@gmail.com and yimmonyneath@gmail.com) if you have any questions.

