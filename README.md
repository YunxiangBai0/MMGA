
# MMGA
<!-- 
Code (pytorch) for ['Source-Free Domain Adaptation via Target Prediction Distribution Search']() on Digits(MNIST, USPS, SVHN), Office-31, Office-Home, VisDA-C, PACS. This paper has been accepted by International Journal of Computer Vision (IJCV). 
DOI: https://doi.org/10.1007/s11263-023-01892-w -->

### Preliminary

- **Datasets**
  - `office-home` [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view)
  - `VISDA-C` [VISDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)
  - `domainnet126` [DomainNet (cleaned)](http://ai.bu.edu/M3SDA/)
You need to download the dataset,  modify the path of images in each '.txt' under the folder './data/'.

The experiments are conducted on one GPU (NVIDIA RTX TITAN).

- python == 3.7.3
- pytorch ==1.6.0
- torchvision == 0.7.0


### Training and evaluation

We provide config files for experiments. 

### Source

- We provide the pre-trained source models which can be downloaded from [here](https://drive.google.com/drive/folders/1nKCKd_hASHbetZBCqWVL2c3ZGyQSL-9p?usp=drive_link).

### Target
After obtaining the source models, modify your source model directory. 

For office-home. 
```bash
CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/sfa_susu/bin/python MMGA_target_oh_vs.py --cls_par 0.3 --da uda --dset office-home --gpu_id 0 --s 0 --t 1 --output_src /media/ts/tntbak2/Modelzoom/source --output ckps/target_mmga/ --seed 2020
```

For VISDA-C.
```bash
CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/sfa_susu/bin/python MMGA_target_oh_vs.py --cls_par 0.2 --da uda --dset office-home --gpu_id 0 --s 0 --t 1 --output_src /media/ts/tntbak2/Modelzoom/source --output ckps/target_mmga/ --net resnet101 --lr 1e-3 --seed 2020
```
For domainnet126. 
```bash
CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/sfa_susu/bin/python MMGA_target_126.py --cls_par 0.3 --da uda --dset domainnet126 --gpu_id 0 --s 0 --t 1 --output_src /media/ts/tntbak2/Modelzoom/source_1024 --output ckps/target_mmga/ --seed 2020
```
You can also  refer to the file on [run.sh](./run.sh).


### Citation
# Tang, S., Chang, A., Zhang, F. et al. Source-Free Domain Adaptation via Target Prediction Distribution Searching. Int J Comput Vis (2023). https://doi.org/10.1007/s11263-023-01892-w

### Acknowledgement


# The code is based on [DeepCluster(ECCV 2018)](https://github.com/facebookresearch/deepcluster) , [SHOT (ICML 2020, also source-free)](https://github.com/tim-learn/SHOT) and [IIC](https://github.com/sebastiani/IIC).


### Contact

- baiyunxiang11@gmail.com
