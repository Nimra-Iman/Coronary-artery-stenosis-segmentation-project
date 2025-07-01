# Coronary-artery-stenosis-segmentation-project
This project presents a deep learning pipeline for the automatic segmentation of coronary artery stenosis in X-ray angiography images using a customized SE-U-Net architecture. It uses a publicly available ARCADE dataset and addresses challenges like class imbalance and limited dataset size with a custom loss function and data augmentation.
## Project Structure:
├── deep learning pipeline/
│   ├── data annotation, preprocessing and augmentation.ipynb
│   └── model architecture and training.ipynb
├── frontend+backend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── model_backend.py
├── stenosis project video.mp4
└── .ipynb_checkpoints/
## SETUP INSTRUCTION:
### Clone the repository
git clone https://github.com/Nimra-Iman/Coronary-artery-stenosis-segmentation-project.git
cd Coronary-artery-stenosis-segmentation-project
### Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
### Run the backend
cd frontend+backend
python model_backend.py
--> Then open index.html in your browser
## Features:
 -> Works on ARCADE X-ray angiography dataset
 -> Custom SE-U-Net architecture to enhance segmentation performance
 -> Custom loss function: Weighted Categorical Crossentropy + Dice Loss to handle class imbalance
 -> Uses Albumentations for extensive data augmentation
 -> Web-based frontend and Flask backend for prediction interface
 -> Demonstration video included
## Dataset:
use ARCADE dataset with VERSION COCO 
-> 1500 X-ray coronary angiography images
-> Annotations in COCO JSON format
-> Train/Val/Test Split
URL: https://zenodo.org/records/10390295
Due to license restrictions, the dataset is not included in this repository. You can request it from the official challenge site or organizers.

