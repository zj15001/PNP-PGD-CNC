PNP-PGD-CNC

The implement of the following paper:
 "Plug-and-Play Algorithm for Magnetic Resonance Image Reconstruction"

Scripts

# PGD-L1

PGD_L1.py (The definition of PGD-L1 algorithm)

# PNP-PGD-L1

PNP_PGD_L1_BM3D.py (The definition is about the use of BM3D denoising under the PNP_PGD_L1 framework)

PNP_PGD_L1_D.py (The definition is about the use of neural network denoising under the PNP_PGD_L1 framework)

# PGD-CNC

PGD_CNC.py (The definition of PGD_CNC algorithm)

# PNP-PGD-CNC

PNP_PGD_CNC_BM3D.py (The definition is about the use of BM3D denoising under the PNP_PGD_CNC framework)

PNP_PGD_CNC_D.py (The definition is about the use of neural network  denoising under the PNP_PGD_CNC framework)

# How to run the scripts?

Run with default settings main.py

All parameters and other required functions are explained in the file "utils/utils.py".

The images used in the experiment are all in the file: testsets/set

The noises and sampling templates used in the experiment are all in the file: CS_MRI

The neural network framework was trained using Zhang Kai, et al. If you want to run this code, please put the download file in the folder ''model_zoo'',
Download link: [https://github.com/cszn/KAIR] ,or download directly from the following link.

*  Google drive download link: [https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D?usp=sharing](https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D?usp=sharing)

*  腾讯微云下载链接: [https://share.weiyun.com/5qO32s3](https://share.weiyun.com/5qO32s3)



# Citation

If you find our code helpful in your resarch or work, please cite our paper.