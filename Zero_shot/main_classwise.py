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

input_pkl_dir = r"D:\Codes\EAV_github\EAV_lab\data_processing"
output_file = "eeg_contrastive_results.txt"
finetuned_models_dir = r"D:\Codes\Finetuned_models_2"

results = list()
def train_all_classes():
    for sub in range(1, 2):
        Data = load_subject_data(directory = input_pkl_dir, subject_idx= sub)
        #tr_x_aud, tr_y_aud, te_x_aud, te_y_aud, tr_x_vis, tr_y_vis, te_x_vis, te_y_vis, tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = Data
        Models = load_models(base_dir = finetuned_models_dir, subject_idx = sub)
        model_aud, model_vis, model_eeg, model_av = Models
        model_zs = ZeroShotModel(eeg_dim = 2600, shared_dim=256, num_classes=5)

        [tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence, te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence] = prepare_multilabel_data(Data[-4:])
        # model_eeg = EEGNet_upd(Chans=30, Samples=500)
        model_eeg = MultiTaskShallowConvNet(Chans=30, Samples=500, num_layers=6)
        # model_eeg = MultiTaskDeepConvNet(Chans=30, Samples=500)
        # model_eeg = MultiTaskTCT(Chans=30, Samples=500)

        trainer = Trainer_eeg_multitask(model = model_eeg, data = [tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence, te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence], num_epochs=400)
        trainer.train()
        out = zeroshot_training(Data=Data, Models=[model_aud, model_vis, model_eeg, model_av], ZeroShotModel=model_zs, epochs=10)
        results.append(out)


results_zs = list()
def train_zero_shot(class_label):
    for sub in range(1, 2):
        Data = load_subject_data(directory=input_pkl_dir, subject_idx=sub)
        Data_zs = prepare_zeroshot_data(Data, exclude_class=class_label)  # excluding chosen class (see 'class labels' description)
        #tr_x_aud, tr_y_aud, te_x_aud, te_y_aud, tr_x_vis, tr_y_vis, te_x_vis, te_y_vis, tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = Data_zs
        [tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence, te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence] = prepare_multilabel_data(Data_zs[-4:])

        Models = load_models(base_dir=finetuned_models_dir, subject_idx=sub)
        model_aud, model_vis, model_eeg, model_av = Models
        model_zs = ZeroShotModel(eeg_dim=2600, shared_dim=256, num_classes=5)

        model_eeg = MultiTaskShallowConvNet(Chans=30, Samples=500, num_layers=6)
        # model_eeg = EEGNet_upd(Chans=30, Samples=500)
        # model_eeg = MultiTaskDeepConvNet2(Chans=30, Samples=500)
        # model_eeg = MultiTaskDeepConvNet(Chans=30, Samples=500)
        trainer = Trainer_eeg_multitask(model=model_eeg,
                                        data=[tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence, te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence], num_epochs=400)
        trainer.train()
        out = zeroshot_training(Data=Data, Models=[model_aud, model_vis, model_eeg, model_av], ZeroShotModel=model_zs, epochs=10) #fusion_model_freeze='unfreeze'
        results_zs.append(out)



#pred_eeg = predict_zeroshot_e_av(Data, Models, model_zs)

#acc_post  = predict_eeg(te_x_av, te_y_av, model_eeg)
#acc = predict_av(te_x_vis, te_x_aud, model_vis, model_aud, model_av, label = te_y_aud)
#print(acc)