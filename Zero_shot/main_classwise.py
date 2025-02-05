from Zeroshot_setting import *
from Transformer_EEG import *
import numpy as np

# for consistent results
import random
def set_random_seed(seed):
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables cuDNN auto-tuning
# Set the random seed
seed = 42  # Replace with any fixed value you want
set_random_seed(seed)

# class labels
'''
'Neutral': 0,
'Happiness': 3,
'Sadness': 1,
'Anger': 2,
'Calmness': 4
'''

# edit these paths based on your folder structure
input_pkl_dir = r"D:\Codes\EAV_github\EAV_lab\data_processing"
finetuned_models_dir = r"D:\Codes\EAV_github\EAV_lab\pretrained_models\Finetuned_models"

results = list()
def train_all_classes():
    # please choose the number of subjects to use
    for sub in range(1, 2):
        Data = load_subject_data(directory = input_pkl_dir, subject_idx= sub)
        Models = load_models(base_dir = finetuned_models_dir, subject_idx = sub)
        [tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence, te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence] = prepare_multilabel_data(Data[-4:])

        model_aud, model_vis, model_av = Models
        model_zs = ZeroShotModel(eeg_dim = 2600, shared_dim=256, num_classes=5)
        # if necessary edit num_layers
        model_eeg = MultiTaskShallowConvNet(Chans=30, Samples=500, num_layers=6)

        # please increase or decrease num_epochs for both models if necessary
        trainer = Trainer_eeg_multitask(model = model_eeg,
                                        data = [tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence, te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence], num_epochs=100)
        trainer.train()
        out = zeroshot_training(Data=Data, Models=[model_aud, model_vis, model_eeg, model_av], ZeroShotModel=model_zs, epochs=5)
        results.append(out)

    #print("Predictions, can be used for further analysis: ", results)

results_zs = list()
def train_zero_shot(class_label):
    # please choose the number of subjects to use
    for sub in range(1, 2):
        Data = load_subject_data(directory=input_pkl_dir, subject_idx=sub)
        Data_zs = prepare_zeroshot_data(Data, exclude_class=class_label)  # excluding chosen class (see 'class labels' description)
        [tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence, te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence] = prepare_multilabel_data(Data_zs[-4:])

        Models = load_models(base_dir=finetuned_models_dir, subject_idx=sub)
        model_aud, model_vis, model_av = Models
        model_zs = ZeroShotModel(eeg_dim=2600, shared_dim=256, num_classes=5)
        # if necessary edit num_layers
        model_eeg = MultiTaskShallowConvNet(Chans=30, Samples=500, num_layers=6)

        # please increase or decrease num_epochs for both models if necessary
        trainer = Trainer_eeg_multitask(model=model_eeg,
                                        data=[tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence, te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence], num_epochs=100)
        trainer.train()
        out = zeroshot_training(Data=Data, Models=[model_aud, model_vis, model_eeg, model_av], ZeroShotModel=model_zs, epochs=5)
        results_zs.append(out)

    #print(results_zs)
