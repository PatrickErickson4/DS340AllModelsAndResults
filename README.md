# 340W Final Model evaluation Repo

install the requirements with requirements.txt.

View the jupyter notebooks to see proof of model training. 
## DO NOT RUN trainAllModels.ipynb. This file is to show that the models were indeed trained by us

## Datasets
---

Datasets include pure 
  - **PlantVillage**: denoted as PlantVillage-Tomato
  - **Tomato-Merged**: PlantVillage injected with images from 6 other datasets. (Novelty)

### Steps to run the Code
### Ensure you are using a Python 3.10.9 Environment. The modules are not compatible with other versions of python.
Ensure you also have pip 25.0.1

**If you wish to use CUDA, use CUDA 11.2 and install cuDNN 8.1 This is highly recommended if you wish to TRAIN any model.**
  - Step 1: Create a folder where you want the file to be.
  - Step 2: Open your command line of choice (powershell, cmd, mac alternatives, linux, etc.)
  - Step 3: Navigate to a projects folder with 
  ```
  cd yourFolderName
  ```
  - Step 4: Clone the repository with 
  ```
  git clone https://github.com/PatrickErickson4/DS340AllModelsAndResults
  ```
  - Step 5: Install the dependencies with 
  ```
  pip install --upgrade --force-reinstall -r requirements.txt
  ```
  - Step 6: Go into exampleTrainer.ipynb to see the formatting of the model training and testing.

Feel free to use evalAllModels.ipynb to verify runs. 

### Models
The following model names in the directory correspond to the following features turned on or off in our ablation study (note all have an extra dense layer):
  - **ExampleModel**: A model that you can play around with in exampleTrainer.ipynb
  - **MobileVITCLAHE**: MobileVIT, trained with CLAHE pre-processing and data augmentations
  - **MobileCLAHENoAug** MobileVIT, trained with only pre-processing and data augmentations
  - **MobileVITDense**: MobileVIT, trained with no augmentations.
  - **MobileVITNoCLAHE**: MobileVIT, trianed with only augmentations.
  - **V2ParentPaperOnNewData**: The parent paper's best model, with CLAHE and augmentations as stated within the paper.
  - **V3LargeCLAHE**: MobileVIT, trained with CLAHE pre-processing and data augmentations
  - **V3LargeCLAHENoAug** MobileVIT, trained with only pre-processing and data augmentations
  - **V3LargeDense**: MobileVIT, trained with no augmentations.
  - **V3LargeNoCLAHE**: MobileVIT, trianed with only augmentations.
  - **V3SmallCLAHE**: MobileVIT, trained with CLAHE pre-processing and data augmentations
  - **V3SmallCLAHENoAug** MobileVIT, trained with only pre-processing and data augmentations
  - **V3SmallVITDense**: MobileVIT, trained with no augmentations.
  - **V3SmallVITNoCLAHE**: MobileVIT, trianed with only augmentations.
### Original Readme
---

This repository is the official implementation of the paper published in IEEE Access Titled 

[S. Ahmed, M. B. Hasan, T. Ahmed, M. R. K. Sony and M. H. Kabir, "Less is More: Lighter and Faster Deep Neural Architecture for Tomato Leaf Disease Classification," ](https://ieeexplore.ieee.org/document/9810234) in IEEE Access Journal. DOI: 10.1109/ACCESS.2022.3187203.

This branch contains the inference code. 



## Citation Instructions
If you use part of the paper or code, please cite this paper with the following bibtex:
```
@article{ahmed2022less,  
        author={Ahmed, Sabbir and Hasan, Md. Bakhtiar and Ahmed, Tasnim and Sony, Md. Redwan Karim and Kabir, Md. Hasanul},  
        journal={{IEEE Access}},   
        title={{Less is More: Lighter and Faster Deep Neural Architecture for Tomato Leaf Disease Classification}},   
        year={2022},  
        volume={},  
        number={},  
        pages={1-1},  
        doi={10.1109/ACCESS.2022.3187203},
        url={https://ieeexplore.ieee.org/document/9810234},
        publisher={{IEEE}}
        }
```

