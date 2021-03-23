# Unsupervised Deep Homography - PyTorch Implementation

[**Unsupervised Deep Homography: A Fast and Robust Homography Estimation
Model**](https://arxiv.org/abs/1709.03966)<br>
Ty Nguyen, Steven W. Chen, Shreyas S. Shivakumar, Camillo J. Taylor, Vijay
Kumar<br>

```bash
cd code/
```
in code/ folder:

`dataset.py`: class SyntheticDataset(torch.utils.data.Dataset) implementation<br>
`homography_model.py`: Unsupervised deep homography model implementation<br>
`homography_CNN_synthetic.py`: Train and test

## Preparing training dataset (synthetic)
Download MS-COCO 2014 dataset<br>
Store Train and test set into RAW_DATA_PATH and TEST_RAW_DATA_PATH respectly.
### Generate training dataset
It will take a few hours to generate 100.000 data samples.
```bash
python utils/gen_synthetic_data.py --mode train
```
### Generate test dataset
```bash 
python utils/gen_synthetic_data.py --mode test 
```

## Train model with synthetic dataset
```bash 
python homography_CNN_synthetic.py --mode train
```

## Test model with synthetic dataset
Download pre-trained weights
```bash 
链接：https://pan.baidu.com/s/102ilb5HJGydpeHtYelx_Xw  提取码：boq9 
```
Store the model to models/synthetic_models folder
```bash 
python homography_CNN_synthetic.py --mode test
```

results | 
---   | 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210322175425747.png?x-oss-process) | 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210322175643842.png?x-oss-process) | 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210322180132270.png?x-oss-process) | 
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021032218020122.png?x-oss-process) | 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210322180502181.png?x-oss-process) | 

##  Reference
[https://github.com/tynguyen/unsupervisedDeepHomographyRAL2018](https://github.com/tynguyen/unsupervisedDeepHomographyRAL2018)
