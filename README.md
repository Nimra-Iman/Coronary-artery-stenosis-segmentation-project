# Coronary-artery-stenosis-segmentation-project
This project presents a deep learning pipeline for the automatic segmentation of coronary artery stenosis in X-ray angiography images using a customized SE-U-Net architecture. It uses a publicly available ARCADE dataset and addresses challenges like class imbalance and limited dataset size with a custom loss function and data augmentation.
## Project Structure:
├── deep learning pipeline/<br>
│   ├── data annotation, preprocessing and augmentation.ipynb<br>
│   └── model architecture and training.ipynb<br>
├── frontend+backend/<br>
│   ├── index.html<br>
│   ├── style.css<br>
│   ├── script.js<br>
│   └── model_backend.py<br>
├── stenosis project video.mp4<br>
└── .ipynb_checkpoints/<br>
## SETUP INSTRUCTION:
### Clone the repository
git clone https://github.com/Nimra-Iman/Coronary-artery-stenosis-segmentation-project.git<br>
cd Coronary-artery-stenosis-segmentation-project<br>
### Create and activate virtual environment
python -m venv venv<br>
source venv/bin/activate  # On Windows: venv\Scripts\activate<br>
pip install -r requirements.txt<br>
### Run the backend
cd frontend+backend<br>
python model_backend.py<br>
--> Then open index.html in your browser<br>
## Features:
 -> Works on ARCADE X-ray angiography dataset<br>
 -> Custom SE-U-Net architecture to enhance segmentation performance<br>
 -> Custom loss function: Weighted Categorical Crossentropy + Dice Loss to handle class imbalance<br>
 -> Uses Albumentations for extensive data augmentation<br>
 -> Web-based frontend and Flask backend for prediction interface<br>
 -> Demonstration video included<br>
## Dataset:
use ARCADE dataset with VERSION COCO <br>
-> 1500 X-ray coronary angiography images<br>
-> Annotations in COCO JSON format<br>
-> Train/Val/Test Split<br>
URL: https://zenodo.org/records/10390295<br>
Due to license restrictions, the dataset is not included in this repository. You can request it from the official challenge site or organizers.

