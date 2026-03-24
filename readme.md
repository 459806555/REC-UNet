REC-UNet

# REC-UNet
REC-UNet: A U-Net model with residual cross-dimensional attention for liver tumor segmentation


## Abstract
Liver tumors impose a significant global health burden, underscoring the urgent need for efficient and accurate diagnostic methods. Computer‑assisted techniques, particularly deep learning‑based segmentation models, have shown considerable promise in this domain. However, they continue to face persistent challenges in liver tumor segmentation, including severe background noise interference, indistinct lesion boundaries, and the difficulty of simultaneously improving segmentation accuracy while maintaining a balanced trade‑off between Recall and Precision. To address these issues, this paper proposes a novel residual “Enhancement‑Calibration” U‑Net architecture, termed REC‑UNet. The model consists of two task‑specific modules: a Residual Enhancement Module (REM) and a Calibration Module (CM). REM leverages residual connections and cross‑dimensional attention to enhance tumor feature representation for accurate segmentation, thereby establishing a foundation for balancing Recall and Precision. CM further mitigates noise propagation from shallow to deep feature layers, refining segmentation precision while sustaining high levels of both Recall and Precision. Experiments on the LiTS2017 and MSD\_Task08 liver tumor datasets demonstrate that REC‑UNet achieves superior performance over mainstream models, with a 4.34\% improvement in Dice and a 4.24\% improvement in IoU over the second-best model (VM-UNet) on LiTS2017. We further validate the model on an in‑house clinical liver tumor MRI dataset, where it attains a Dice score of 87.88\% and an IoU of 86.91\%, while maintaining a well‑balanced trade‑off between Recall and Precision. Importantly, REC-UNet achieves high overall segmentation accuracy across diverse lesion sizes and contrast conditions without relying on explicit size-stratified optimization. These results confirm the robust generalizability of REC‑UNet and highlight its significant clinical value for computer‑assisted liver tumor diagnosis.
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


