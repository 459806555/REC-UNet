REC-UNet

# REC-UNet
REC-UNet: A Residual Attention-Based U-Net Model for Enhancing Liver Tumor Segmentation


## Abstract
Liver tumors present a significant global health challenge, necessitating efficient diagnostic methods.  Computer-assisted techniques, particularly deep learning models, offer promise but face challenges due to noise interference and the segmentation of small lesions in early-stage liver cancer.  To address these issues, we introduce REC-UNet, an 'enhancement-calibration' U-shaped network incorporating a Residual Enhancement Module (REM) and a Calibration Module (CM).  These modules enhance attention to small target tumors and mitigate the impact of noise.  Experiments on the LiTS2017 and MSD\_Task08 liver tumor datasets demonstrate REC-UNet's superior performance over mainstream models, achieving a 4.34\% improvement in Dice score and a 4.24\% improvement in IoU score compared to the second-best model on the LiTS2017 dataset.  These results indicate REC-UNet's potential for early tiny tumor prediction and its robustness and generalizability.
## 0. Main Environments
joblib==1.3.2
numpy==1.23.2
numpy==1.24.4
opencv_python==4.4.0.44
pandas==1.2.4
Pillow==10.0.1
Pillow==8.2.0
Pillow==11.3.0
scikit_learn==1.3.2
scipy==1.10.1
skimage==0.0
torch==1.9.0+cu111
torchvision==0.10.0+cu111
tqdm==4.66.1

## 1. Prepare the dataset
- The processed LiTS2017 and MSD_task08 datasets can be found here {[alipan](https://www.alipan.com/s/ADtHQrgxRyM)}. 

- After downloading the datasets, you are supposed to put them into './LiTS_data/' and './MSD_data/', and the file format reference is as follows.

- './LiTS_data/'
  - train_image
      - .npy
   - train_label
      - .npy
  - test_image
      - .npy
   - test_label
      - .npy

- './MSD_data/'
  - train_image
      - .npy
   - train_label
      - .npy
  - test_image
      - .npy
   - test_label
      - .npy
## 2. Trained weights

- 
Trained weights can be downloaded from [Baidu] https://pan.baidu.com/s/1SzAAboONChXliX2mClvoCg?pwd=26vm

## 3. Train the REC-UNet
3.1 LiTS_tumor
```bash
cd REC-UNet
python LiTS_train.py --epochs 100 --lr 0.0003
```
After training, please manually load the last saved .pth file at line 261 of LiTS_train.py.
```bash
python LiTS_train.py --epochs 20 --lr 0.0001
```
3.2 LiTS_liver
At line 258 of LiTS_train.py, manually change the model to: model = REC_UNet_EM.REC_UNet(args)
```bash
cd REC-UNet
python LiTS_train.py --epochs 100 --lr 0.0003
```
After training, please manually load the last saved .pth file at line 261 of LiTS_train.py.
```bash
python LiTS_train.py --epochs 20 --lr 0.0001
```
3.2 MSD_Task08
```bash
cd REC-UNet
python MSD08_train.py --epochs 200 --lr 0.0003
```



