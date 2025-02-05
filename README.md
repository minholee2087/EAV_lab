
# **Multimodal EEG, Audio, and Vision for Emotion Recognition** (Temporary Draft)

This repository contains two main directories:  
- **`Zero_shot/`**: Implements the Zero-Shot EEG-based emotion classification model as described in *"Multimodal Joint Representations of EEG and Audio-Vision for Zero-Shot Learning"* (under review).  
- **`EAV_Fusion/`**: Apply feature fusion technique to enhance multimodal emotion recognition as described in *"Adaptive Bottleneck Transformer for Multimodal EEG, Audio, and Vision Fusion"* (under review).  

## **Multimodal Joint Representations of EEG and Audio-Vision for Zero-Shot Learning**  
**Overview:**  This repository contains the implementation of our multimodal zero-shot learning (ZSL) framework for EEG-based emotion recognition, as presented in the paper (currently under review): "Multimodal Joint Representations of EEG and Audio-Vision for Zero-Shot Learning". Our approach integrates EEG, audio, and vision modalities to map data into a shared semantic embedding space using contrastive learning. The framework leverages a multimodal audio-vision transformer alongside a shallow EEG transformer to optimize both unimodal and multimodal performance.

## **Adaptive Bottleneck Transformer for Multimodal EEG, Audio, and Vision Fusion (AMBT)**  

**Overview:**  This repository contains the implementation of the Adaptive Multimodal Bottleneck Transformer (AMBT), a novel architecture designed for efficient multimodal fusion of EEG, Audio, and Vision data in emotion recognition tasks. AMBT includes two versions: **AMBT-Mean**, which applies mean-based fusion of bottleneck tokens for multimodal integration, and **AMBT-Concat**, which utilizes concatenation-based fusion. Each modality—EEG, Audio, and Vision—is processed by its own dedicated Transformer model, ensuring optimal feature extraction. Through cross-modal learning, AMBT maintains unimodal processing pipelines while enabling stronger modalities to extract meaningful signals from weaker ones using implicit contrastive learning. Extensive experiments on the EAV (EEG-Audio-Vision) benchmark dataset demonstrate state-of-the-art performance in multimodal fusion.

---

## **📌 Dataset Description**  
### **EAV: EEG-Audio-Video Dataset**  
A multimodal emotion dataset comprising data from 30-channel electroencephalography (EEG), audio, and video recordings from 42 participants. Each participant engaged in a cue-based conversation scenario, eliciting five distinct emotions:  
- Neutral (N)
- Anger (A)  
- Happiness (H)  
- Sadness (S)  
- Calmness (C)  

Participants engage in paired listen/speak sets with recordings of an experienced actor.  
Throughout the experiment, each participant contributed 200 interactions, resulting in a cumulative total of 8,400 interactions across all participants.  

📄 For more details, refer to the dataset paper and its GitHub repository:  
🔗 [https://www.nature.com/articles/s41597-024-03838-4](https://www.nature.com/articles/s41597-024-03838-4)  
🔗 [https://github.com/nubcico/EAV](https://github.com/nubcico/EAV)  


---

## **⚙️ Setup and Installation**  
### **1. Downloading the Data**  

Follow this link for instructions to download the dataset:  
🔗 [https://github.com/nubcico/EAV](https://github.com/nubcico/EAV)  

If you want to run an executable code on 1 subject, please follow this link (it also contains large files from **'pretrained_models\'**):  
🔗 [https://drive.google.com/drive/folders/13tGH7TJEtokCIZo1hQF0MgueHSN3fqwa?usp=sharing](https://drive.google.com/drive/folders/data_input)

### **2. Running the Model**  

Install the required dependencies:  
```bash
pip install -r requirements.txt
```

For **Zero-Shot Learning**, execute the main script:  
```bash
python Zero_shot/main.py
```

For **EAV Fusion**, you can choose from two versions:  
```bash
python Fusion_bottleneck/AMBT_mean.py
```
or  
```bash
python Fusion_bottleneck/AMBT_concat.py
```

---


## **📢 Citation**  
If you use this code and dataset, please cite:  
- This GitHub repository: [https://github.com/minholee2087/EAV_lab](https://github.com/minholee2087/EAV_lab)  
- The **EAV dataset**: [https://github.com/nubcico/EAV](https://github.com/nubcico/EAV)  

---

### **⚠️ Note**  
🚧 **This GitHub repository is a temporary draft.** 🚧  

---

## **📁 Repository Structure**  
├── Zero_shot/                     # Zero-Shot EEG classification setup  
│   ├── Transformer_Audio.py       # Transformer model for processing audio modality  
│   ├── Transformer_EEG.py         # Transformer model for processing EEG modality  
│   ├── Transformer_Video.py       # Transformer model for processing video modality  
│   ├── Zeroshot_setting.py        # Configuration and setup for zero-shot learning experiments  
│   ├── main_classwise.py          # Script for evaluating zero-shot learning on a class-wise basis  
│   ├── main_classwise_visualization.py  # Script for visualizing class-wise zero-shot results  
│   ├── main.py                    # Main executable script for running zero-shot learning experiments  
│   ├── utils.py                   # Utility functions for data processing and evaluation  
│   ├── config.yaml                # Configuration file containing model hyperparameters  
│   ├── results                    # Directory storing model outputs, logs, and evaluations  
├── EAV_Fusion/              # Adaptive Multimodal Bottleneck Transformer (AMBT) models  
│   ├── Transformer_Audio_mean.py   # Transformer model for audio modality (AMBT-Mean)  
│   ├── Transformer_EEG_mean.py     # Transformer model for EEG modality (AMBT-Mean)  
│   ├── Transformer_Video_mean.py   # Transformer model for video modality (AMBT-Mean)  
│   ├── Transformer_Audio_concat.py # Transformer model for audio modality (AMBT-Concat)  
│   ├── Transformer_EEG_concat.py   # Transformer model for EEG modality (AMBT-Concat)  
│   ├── Transformer_Video_concat.py # Transformer model for video modality (AMBT-Concat)  
│   ├── AMBT_mean.py                # Implementation of AMBT-Mean fusion architecture  
│   ├── AMBT_concat.py              # Implementation of AMBT-Concat fusion architecture  
├── pretrained_models              # Directory for storing pre-trained model checkpoints  
├── data_processing                # Scripts for preprocessing the EEG-Audio-Vision (EAV) dataset  
├── requirements.txt                # List of required dependencies for the project  
├── README.md                       # Project documentation and usage instructions  
└── LICENSE                         # License information for the repository  
