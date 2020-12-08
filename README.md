# NSHE
Source code of "IJCAI20 - Network Schema Preserving Heterogeneous Information Network Embedding"
The paper is publically available at http://shichuan.org/doc/87.pdf

---
# Requirements
Python >=3.5
Pytorch >= 1.0

# Others
Please note that,
- The sampling process in each epoch is extremely time consuming, the sampled results are saved as temp files for different seeds to skip sampling and load the temp files instead.
- We repeated the experiments with 10 different seeds and reported the mean result.

# BibTex
If you find our work useful, please cite our paper:

@inproceedings{ijcai2020-190,
  title     = {Network Schema Preserving Heterogeneous Information Network Embedding},
  author    = {Zhao, Jianan and Wang, Xiao and Shi, Chuan and Liu, Zekuan and Ye, Yanfang},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere}	
  pages     = {1366--1372},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/190},
  url       = {https://doi.org/10.24963/ijcai.2020/190},
}
