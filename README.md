# SocialMedia-TextImage-Classification-AuxLosses
A collection of multimodal models enhanced with auxiliary losses designed specifically for social media text-image classification tasks. 


D. Sanchez Villegas, D. Preoțiuc-Pietro, and N. Aletras (2024). Improving Multimodal Classification of Social Media Posts by Leveraging Image-Text Auxiliary Tasks. EACL Findings 2024.


Multimodal fusion using text-image relationship

1. Create a new environment
```conda env create -f timrel-env.yml```
2. Create, set the data directory and upload data files including the data_key_*.csv files and the image files.

   DATA DIRECTORY
    - The data directory path can be modified in the models/config.py
   ```DATA_PATH = "SocialMedia-TextImage-Classification-AuxLosses/data/"```

   DATA KEY FILES
    - The data_key_*.csv should be placed in DATA_PATH. Dummy examples are in [SocialMedia-TextImage-Classification-AuxLosses/data](https://github.com/danaesavi/ocialMedia-TextImage-Classification-AuxLosses/tree/main/data).

   IMAGES
    - Image examples can be found in SocialMedia-TextImage-Classification-AuxLosses/data/MSD/dataset_image (sarcasm detection) and ocialMedia-TextImage-Classification-AuxLosses/data/MVSA-Single/data (sentiment analysis)
    - They should be placed in DATA_PATH+'data/MSD/dataset_image' and DATA_PATH+'MVSA-Single/data' respectively  

   Datasets:
    - [TIR](https://github.com/danielpreotiuc/text-image-relationship/) [Paper](https://aclanthology.org/P19-1272.pdf)
    - [MVSA](http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/) | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-27674-8_2)
    - [MHP](https://zenodo.org/record/5123567#.Y-OAq-zP1pR) | [Paper](https://aclanthology.org/2021.findings-acl.166.pdf)
    - [MSD](https://github.com/headacheboy/data-of-multimodal-sarcasm-detection) | [Paper](https://aclanthology.org/P19-1239.pdf) 
    - [MICD](https://github.com/danaesavi/micd-influencer-content-twitter) | [Paper](http://www.afnlp.org/conferences/ijcnlp2023/proceedings/main-long/cdrom/pdf/2023.ijcnlp-long.15.pdf)
   

Example of how to run Ber-ViT-Att (testing: uses a sample of 100 data points of the corresponding dataset)

```
module load Anaconda3
module load CUDA
source activate timrel-env
python3 run_mm_late.py --txt_model_name bernice --img_model_name vit --fusion_name attention --task 2 --epochs 7 --seed 40 --testing
```


