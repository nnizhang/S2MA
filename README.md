# S2MA
source code for our CVPR 2020 paper “Learning Selective Self-Mutual Attention for RGB-D Saliency Detection” by Nian Liu, Ni Zhang  and Junwei Han.

created by Ni Zhang, email: nnizhang.1995@gmail.com

## Usage

### Requirement
1. pytorch 0.4.1
2. torchvision 0.1.8

### Training
1. download the RBD-D datasets [[baidu pan](https://pan.baidu.com/s/1q4g9n_n4X_b4WbrhiFuxOw) fetch code: chdz | [Google drive](https://drive.google.com/drive/folders/1ZKK7Le5veXJVD3DZ8OdrO9CdqL2QOFAl?usp=sharing)] and pretrained VGG model [[baidu pan](https://pan.baidu.com/s/19cik8v7Ix5YOo7sdEosp9A) fetch code: dyt4 | [Google drive](https://drive.google.com/drive/folders/1ZKK7Le5veXJVD3DZ8OdrO9CdqL2QOFAl?usp=sharing)], then put them in the ./RGBdDataset_processed directory and ./pretrained_model directory, respectively.
2. run `python generate_list.py` to generate the image lists.
3. modify codes in the parameter.py
4. start to train with `python train.py`


### Testing
1. download our models [[baidu pan](https://pan.baidu.com/s/16hfdk-yE5-sy9B9v6oT1oQ) fetch code: ly9k | [Google drive](https://drive.google.com/drive/folders/1ZKK7Le5veXJVD3DZ8OdrO9CdqL2QOFAl?usp=sharing)] and put them in the ./models directory. After downloading, you can find two models (S2MA.pth and S2MA_DUT.pth). S2MA_DUT.pth is used for testing on the DUT-RGBD dataset and S2MA.pth is used for testing on the rest datasets.
2. modify codes in the parameter.py
3. start to test with `python test.py` and the saliency maps will be generated in the ./output directory.

Our saliency maps can be download from [[baidu pan](https://pan.baidu.com/s/1G-M18V7taJZb44awqxg4tw) fetch code: frzb | [Google drive](https://drive.google.com/drive/folders/1ZKK7Le5veXJVD3DZ8OdrO9CdqL2QOFAl?usp=sharing)].

## Acknowledgement
We use some opensource codes from [Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch), [denseASPP](https://github.com/DeepMotionAIResearch/DenseASPP). Thanks for the authors.

## Citing our work
If you think our work is helpful, please cite 
```
@inproceedings{liu2020S2MA, 
  title={Learning Selective Self-Mutual Attention for RGB-D Saliency Detection}, 
  author={Liu, Nian and Zhang, Ni and Han, Junwei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}

