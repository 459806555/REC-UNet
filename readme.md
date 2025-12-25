REC-UNet

# REC-UNet
REC-UNet: A U-Net model with residual cross-dimensional attention for liver tumor segmentation


## Abstract
Liver tumors present a significant global health challenge, necessitating efficient diagnostic methods. Computer-assisted techniques, particularly deep learning models, hold promise but encounter challenges in liver cancer lesion segmentation due to issues such as severe noise interference, low contrast, and poor boundary distinguishability in medical images. To address these issues, this paper proposes a residual "Enhancement-Calibration" U-shaped network, named REC-UNet. The model incorporates two novel modules: a Residual Enhancement Module (REM) and a Calibration Module (CM). The REM leverages residual connections and cross-dimensional attention to achieve multidimensional perception of tumor morphology within complex backgrounds. Simultaneously, the CM weakens the propagation of noise from shallow features to deeper layers, thereby enhancing the model's robustness. Experiments on the LiTS2017 and MSD_Task08 liver tumor datasets demonstrate REC-UNet's superior performance over mainstream models, achieving a 4.34% improvement in Dice score and a 4.24% improvement in IoU score compared to the second-best model on the LiTS2017 dataset. Subsequently, we performed additional tests on the hospital clinical liver tumor MRI dataset, achieving Dice and IoU scores of 87.88% and 86.91%, respectively. The results underscore the value of REC-UNet for the clinical prediction of liver tumors, confirming its robustness and strong generalizability. Our code is available at https://github.com/459806555/REC-UNet.
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
3.3 Clinical Liver Tumor MRI Dataset
The clinical dataset adopts the same training and testing pipelines as well as parameters as the MSD_Task08 dataset, with the only difference being the number of channels. This is because the clinical dataset only has a single-channel mask for tumors, while MSD_Task08 features a two-channel mask (covering both hepatic vessels and hepatic tumors).
The clinical datasets are not publicly available due to ethical and privacy restrictions.


