# S3R-Net: A Single-Stage Approach to Self-Supervised Shadow Removal (NTIRE @ CVPR 2024) - Kubiak _et al._
Project repo for the paper [S3R-Net: A Single-Stage Approach to Self-Supervised Shadow Removal](https://arxiv.org/pdf/2404.12103)

## The basics
The model was developed in an env based on Pytorch 1.8.1 with CUDA 11.1 (docker image: nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04). Key dependancies: ```kornia```


## Testing
First, download the checkpoints ([ISTD](https://personalpages.surrey.ac.uk/s.hadfield/istd_ckpt.pth) | [AISTD](https://personalpages.surrey.ac.uk/s.hadfield/aistd_cktp.pth)) and put them in the relevant folders: ```checkpoints/best_(a)istd```. To run the test, run

## Training
coming soon

## Citation
If you use or write about S3R-Net, please use the below citation:
```
@inproceedings{kubiak_2024_s3rnet,
  title={S3R-Net: A Single-Stage Approach to Self-Supervised Shadow Removal},
  author={Nikolina Kubiak and Armin Mustafa and Graeme Phillipson and Stephen Jolly and Simon Hadfield},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2024}
}
```
