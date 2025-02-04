This repository contains two main directories: 'Zero_shot/', which implements the Zero-Shot EEG-based emotion classification model as described in 'Multimodal Joint Representations of EEG and Audio-Vision for Zero-Shot Learning' (under review), and 'Fusion_bottleneck/', which explores feature fusion techniques to enhance multimodal emotion recognition.

## Multimodal Joint Representations of EEG and Audio-Vision for Zero-Shot Learning  
**Overview**
This repository contains the implementation of our multimodal zero-shot learning (ZSL) framework for EEG-based emotion recognition, as presented in the paper (currently under review): "Multimodal Joint Representations of EEG and Audio-Vision for Zero-Shot Learning".
Our approach integrates EEG, audio, and vision modalities to map data into a shared semantic embedding space using contrastive learning. The framework leverages a multimodal audio-vision transformer alongside a shallow EEG transformer to optimize both unimodal and multimodal performance.

## Fusion Bottleneck paper
**Overview**
This repository also contains the implementation of...

## **📁 Repository Structure**  
├── Zero_shot/                     # Contains the Zero-Shot EEG classification model  
│   ├── Transformer_Audio.py       # Transformer model for audio modality  
│   ├── Transformer_EEG.py         # Transformer model for EEG modality  
│   ├── Transformer_Video.py       # Transformer model for video modality  
│   ├── Zeroshot_setting.py        # Configuration and setup for zero-shot learning  
│   ├── main_classwise.py          # Main script for class-wise evaluation  
│   ├── main_classwise_visualization.py  # Visualization of class-wise results  
│   ├── main.py                    # Main executable script for ZSL (runs the chosen settings)  
│   ├── utils.py                    # Helper functions  
│   ├── config.yaml                 # Model configurations  
│   ├── results/                    # Stores model outputs and logs  
├── Fusion_bottleneck/              # Implements feature fusion techniques for emotion recognition  
├── pretrained_models/              # Pre-trained models for users to download and fine-tune  
├── data_processing/                # Scripts for preprocessing the EAV (EEG-Audio-Vision) dataset  
│   ├── EEG/                        # Contains EEG data files  
│   │   ├── subject_01.pkl          # Example EEG data file  
│   ├── Audio/                      # Contains Audio data files  
│   ├── Video/                      # Contains Video data files  
├── requirements.txt                # List of dependencies  
├── README.md                       # Project documentation  
└── LICENSE                         # License information  


## **⚙️ Setup and Installation**  
1. Downloading the Data
  The EAV dataset is required for training and evaluation. Follow the instructions here to request access and download the dataset.

2. Running the Model
  Install the required dependencies:
    pip install -r requirements.txt
  Execute the main script:
    python main.py

3. Updating Pretrained Models
  To update the pretrained models, download them from this link and place them in the 'pretrained_models/' directory.

Citation
If you use this code, please cite this github 'https://github.com/minholee2087/EAV_lab':

Original GitHub Repository for EAV dataset
This project is based on prior research and implementations. You can find the original repository here: 'https://github.com/nubcico/EAV'.
