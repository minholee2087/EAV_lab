from Transformer_torch_pretrained.Video_Transformer.Zeroshot_setting_ import load_subject_data, load_models, predict_av
from Transformer_torch_pretrained.Video_Transformer.Zeroshot_setting_ import *

#r"C:\Users\minho.lee\Dropbox\Datasets\EAV\Input_images"
#r"C:\Users\minho.lee\Dropbox\Datasets\EAV\Finetuned_models"
#r"D:\Dropbox\DATASETS\EAV\Input_images"
#r"D:\Dropbox\DATASETS\EAV\Finetuned_models"

'''
'Neutral': 0,
'Happiness': 3,
'Sadness': 1,
'Anger': 2,
'Calmness': 4
'''

output_file = "eeg_contrastive_results.txt"
for sub in range(1, 43):
    Data = load_subject_data(directory = r"D:\Dropbox\DATASETS\EAV\Input_images", subject_idx= sub)
    tr_x_aud, tr_y_aud, te_x_aud, te_y_aud, tr_x_vis, tr_y_vis, te_x_vis, te_y_vis, tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = Data
    Models = load_models(base_dir = r"D:\Dropbox\DATASETS\EAV\Finetuned_models", subject_idx = sub)
    model_aud, model_vis, model_eeg, model_av = Models
    model_zs = ZeroShotModel(eeg_dim = 2600, shared_dim=256, num_classes=5)
    ############################################################################

    Data_zs = prepare_zeroshot_data(Data, exclude_class=1) #removed from only training

    zeroshot_training(Data_zs, Models, model_zs, epochs=10)

    zeroshot_training(Data, Models, model_zs, epochs=5, fusion_model_freeze='unfreeze')

    pred_eeg = predict_zeroshot_e_av(Data, Models, model_zs)

    #acc_post  = predict_eeg(te_x_av, te_y_av, model_eeg)
    #acc = predict_av(te_x_vis, te_x_aud, model_vis, model_aud, model_av, label = te_y_aud)
    #print(acc)

