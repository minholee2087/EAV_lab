from Contrastive_learning.Transformer_EEG import *
from Contrastive_learning.Transformer_EEG import ShallowConvNet1
from Contrastive_learning.Zeroshot_setting import *
#r"C:\Users\minho.lee\Dropbox\Datasets\EAV\Input_images"
#r"C:\Users\minho.lee\Dropbox\Datasets\EAV\Finetuned_models"
#r"D:\Dropbox\DATASETS\EAV\Input_images"
#r"D:\Dropbox\DATASETS\EAV\Finetuned_models"
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

import os

def visualize_tsne(features, labels, title="t-SNE Visualization", unseen_classes=None, fname="tsne_plot"):

    class_labels = {
        0: 'Neutral',
        1: 'Sadness',
        2: 'Anger',
        3: 'Happiness',
        4: 'Calmness'
    }
    # Assigning emotion-appropriate colors
    custom_palette = {
        0: "gray",      # Neutral - Gray
        1: "blue",      # Sadness - Blue
        2: "orange",    # Anger - Orange
        3: "red",       # Happiness - Red
        4: "green"      # Calmness - Green
    }

    # Convert tensors to NumPy arrays if necessary
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Assign marker styles and sizes, highlight unseen class
    marker_styles = ['o'] * len(class_labels)  # Default to circles
    marker_sizes = [70] * len(class_labels)   # Default marker size
    if unseen_classes is not None:
        for unseen_class in unseen_classes:
            marker_styles[unseen_class] = '*'  # Use '*' for unseen classes
            marker_sizes[unseen_class] = 170  # Larger size for unseen classes


    # Plotting
    plt.figure(figsize=(10, 8))
    for label, color in custom_palette.items():
        mask = labels == label
        marker = marker_styles[label]
        size = marker_sizes[label]
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=color,
            label=f"{class_labels[label]} ({label})",
            marker=marker,
            s=size,       # Adjust marker size
            alpha=0.7
        )

    # Add legend
    plt.legend(title="Classes")
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(False)

    # Save plot with given filename
    filename = f"{fname}.svg"
    plt.savefig(filename, format="svg")
    print(f"Saved plot as {filename}")




results_path = "Contrastive_results.txt"
results = list()
for sub in range(2, 3):
    Data = load_subject_data(directory = r"D:\Dropbox\DATASETS\EAV\Input_images", subject_idx= sub,  audio=True, vision=True, eeg=True)
    tr_x_aud, tr_y_aud, te_x_aud, te_y_aud, tr_x_vis, tr_y_vis, te_x_vis, te_y_vis, tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = Data
    Models = load_models(base_dir = r"D:\Dropbox\DATASETS\EAV\Finetuned_models", subject_idx = sub)
    model_aud, model_vis, hello, model_av = Models
    model_zs = ZeroShotModel(eeg_dim = 2600, shared_dim=256, num_classes=5)
    ############################################################################

    # contrastive Learning - all classes
    '''
    [tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence, te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence] = prepare_multilabel_data(Data[-4:])
    model_eeg = MultiTaskShallowConvNet(Chans=30, Samples=500, num_layers=6)

    trainer = Trainer_eeg_multitask(model = model_eeg, data = [tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence, te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence], num_epochs=150)
    trainer.train()
    out = zeroshot_training(Data=Data, Models=[model_aud, model_vis, model_eeg, model_av], ZeroShotModel=model_zs, epochs=10)
    results.append(out)

    visualize_tsne(features=model_eeg.feature(tr_x_eeg), labels=tr_y_eeg, fname = 'tsne_eeg_all_tr')
    visualize_tsne(features=model_eeg.feature(te_x_eeg), labels=te_y_eeg, fname = 'tsne_eeg_all_te')

    visualize_tsne(features=model_zs.forward_eeg(model_eeg.feature(tr_x_eeg))[0], labels=tr_y_eeg,  fname = 'tsne_eeg_all_tr_cent')
    visualize_tsne(features=model_zs.forward_eeg(model_eeg.feature(te_x_eeg))[0], labels=te_y_eeg,  fname = 'tsne_eeg_all_te_cent')
    '''

    # Zeroshot Learning - 1 unseen class : H, S
    Data_zs = prepare_zeroshot_data(Data, exclude_class=1)    # removed from only training
    Data_zs = prepare_zeroshot_data(Data_zs, exclude_class=3)
    tr_x_aud, tr_y_aud, te_x_aud, te_y_aud, tr_x_vis, tr_y_vis, te_x_vis, te_y_vis, tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = Data_zs
    [tr_x_eeg_zs, tr_y_eeg, tr_y_arousal, tr_y_valence, te_x_eeg_zs, te_y_eeg, te_y_arousal, te_y_valence] = prepare_multilabel_data(Data_zs[-4:])
    model_eeg = MultiTaskShallowConvNet(Chans=30, Samples=100, num_layers=6)
    trainer = Trainer_eeg_multitask(model = model_eeg, data = [tr_x_eeg_zs, tr_y_eeg, tr_y_arousal, tr_y_valence, te_x_eeg_zs, te_y_eeg, te_y_arousal, te_y_valence], num_epochs=150)
    trainer.train()
    out = zeroshot_training(Data=Data, Models=[model_aud, model_vis, model_eeg, model_av], ZeroShotModel=model_zs, epochs=7)
    results.append(out)


    tr_x_aud, tr_y_aud, te_x_aud, te_y_aud, tr_x_vis, tr_y_vis, te_x_vis, te_y_vis, tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = Data
    visualize_tsne(features=model_eeg.feature(tr_x_eeg), labels=tr_y_eeg, unseen_classes= [1, 3], fname = 'tsne_eeg_zs_HS_tr')
    visualize_tsne(features=model_eeg.feature(te_x_eeg), labels=te_y_eeg, unseen_classes= [1, 3],  fname = 'tsne_eeg_zs_HS_te')

    visualize_tsne(features=model_zs.forward_eeg(model_eeg.feature(tr_x_eeg))[0], labels=tr_y_eeg, unseen_classes= [1, 3],  fname = 'tsne_eeg_zs_HS_tr_cent')
    visualize_tsne(features=model_zs.forward_eeg(model_eeg.feature(te_x_eeg))[0], labels=te_y_eeg, unseen_classes= [1, 3], fname = 'tsne_eeg_ZS_HS_te_cent')

    a = 3


# pretrained, compare with all subject



#acc_post  = predict_eeg(te_x_av, te_y_av, model_eeg)
#acc = predict_av(te_x_vis, te_x_aud, model_vis, model_aud, model_av, label = te_y_aud)
#print(acc)
# Data_zs = prepare_zeroshot_data(Data, exclude_class=1) #removed from only training
# pred_eeg = predict_zeroshot_e_av(Data, Models, model_zs)

'''
'Neutral': 0,
'Sadness': 1,
'Anger': 2,
'Happiness': 3,
'Calmness': 4
'''


'''

subject_indices = [2, 4, 17, 20, 22, 33, 42]

# Initialize lists to store data
all_tr_x = []
all_tr_y = []
all_te_x = []
all_te_y = []
all_tr_arousal = []
all_tr_valence = []
all_te_arousal = []
all_te_valence = []
directory = r"D:\Dropbox\DATASETS\EAV\Input_images"

for subject_idx in subject_indices:
    # Load data for the current subject
    Data = load_subject_data(directory=directory, subject_idx=subject_idx, audio=False, vision=False, eeg=True)

    # Extract EEG data and prepare multilabel data
    [tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence, te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence] = prepare_multilabel_data(Data[-4:])

    # Append training and testing data and labels
    all_tr_x.append(tr_x_eeg)
    all_tr_y.append(tr_y_eeg)
    all_tr_arousal.append(tr_y_arousal)
    all_tr_valence.append(tr_y_valence)
    all_te_x.append(te_x_eeg)
    all_te_y.append(te_y_eeg)
    all_te_arousal.append(te_y_arousal)
    all_te_valence.append(te_y_valence)

# Concatenate all training and testing data and labels
tr_x_combined = torch.cat(all_tr_x, dim=0)
tr_y_combined = torch.cat(all_tr_y, dim=0)  # Emotion labels
tr_arousal_combined = torch.cat(all_tr_arousal, dim=0)  # Arousal labels
tr_valence_combined = torch.cat(all_tr_valence, dim=0)  # Valence labels
te_x_combined = torch.cat(all_te_x, dim=0)
te_y_combined = torch.cat(all_te_y, dim=0)  # Emotion labels
te_arousal_combined = torch.cat(all_te_arousal, dim=0)  # Arousal labels
te_valence_combined = torch.cat(all_te_valence, dim=0)  # Valence labels

# Update the combined labels for multitask training
tr_y_multilabel = [tr_y_combined, tr_arousal_combined, tr_valence_combined]
te_y_multilabel = [te_y_combined, te_arousal_combined, te_valence_combined]

# Initialize model and trainer
model_eeg_subs = MultiTaskShallowConvNet(Chans=30, Samples=500, num_layers=6)
trainer = Trainer_eeg_multitask(
    model=model_eeg_subs,
    data=[tr_x_combined, tr_y_combined, tr_arousal_combined, tr_valence_combined,
          te_x_combined, te_y_combined, te_arousal_combined, te_valence_combined],
    num_epochs=400
)
trainer.train()
'''

