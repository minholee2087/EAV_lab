from Zeroshot_setting import *
from Transformer_EEG import *
import numpy as np
import torch
import gc

# weights randomness for consistent results
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

# Check if GPU memory is available
def check_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"GPU Memory Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.2f} GB")

# clear the GPU memory - to avoid any errors related to "memory allocation"
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Add memory management function
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# class labels - change the exclude class in Zero-shot setting based on the following labels
'''
'Neutral': 0,
'Happiness': 3,
'Sadness': 1,
'Anger': 2,
'Calmness': 4
'''

# IMPORTANT: update these paths based on your folder structure
input_pkl_dir = r"D:\Codes\Github\EAV_lab\Zero_shot\data_input\data_processing"
finetuned_models_dir = r"D:\Codes\Github\EAV_lab\Zero_shot\data_input\pretrained_models\Finetuned_models"
results_path = r"D:\Codes\Github\EAV_lab\Zero_shot\results"

results = list()
def train_all_classes():
    # please choose the number of subjects to use (from 1 to 43) - overall 42 subjects (but here we have only 1 for demo)
    for sub in range(1, 2):
        try:
            # Clear memory at the start of each iteration - to prevent potential issues
            clear_memory()

            print(f"Processing subject {sub} ...")
            Data = load_subject_data(directory = input_pkl_dir, subject_idx= sub)
            Models = load_models(base_dir = finetuned_models_dir, subject_idx = sub)
            [tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence,
             te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence] = prepare_multilabel_data(Data[-4:])

            model_aud, model_vis, model_av = Models
            model_zs = ZeroShotModel(eeg_dim = 2600, shared_dim=256, num_classes=5)
            # if necessary edit num_layers
            model_eeg = MultiTaskShallowConvNet(Chans=30, Samples=500, num_layers=2)
            # please increase or decrease num_epochs for both models if necessary
            trainer = Trainer_eeg_multitask(model = model_eeg,
                                            data = [tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence,
                                                    te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence], num_epochs=500)
            trainer.train()
            out = zeroshot_training(Data=Data, Models=[model_aud, model_vis, model_eeg, model_av],
                                    ZeroShotModel=model_zs, epochs=100)
            results.append(out)

            # Clear memory after each subject
            del model_zs, model_eeg, trainer, Data, Models
            clear_memory()

        except RuntimeError as e:
            print(f"Error processing subject {sub}: {e}")
            clear_memory()
            continue
        except Exception as e:
            print(f"Unexpected error processing subject {sub}: {e}")
            clear_memory()
            continue

    # results will contain predicted labels for all models, and includes true labels as well
    # see Zeroshot_setting.zeroshot_training() return for more details
    #print("Predictions, can be used for further analysis: ", results)
    with open(results_path+"/allclasses_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("✅ Results saved to 'allclasses_results.pkl'")

results_zs = list()
def train_zero_shot(class_label):
    # please choose the number of subjects to use (from 1 to 43) - overall 42 subjects (but here we have only 1 for demo)
    for sub in range(1, 2):
        try:
            # Clear memory at the start of each iteration - to prevent potential issues
            clear_memory()

            print(f"Processing subject {sub} ...")
            Data = load_subject_data(directory=input_pkl_dir, subject_idx=sub)
            Data_zs = prepare_zeroshot_data(Data, exclude_class=class_label)  # excluding chosen class (see 'class labels' description)
            [tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence,
             te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence] = prepare_multilabel_data(Data_zs[-4:])

            Models = load_models(base_dir=finetuned_models_dir, subject_idx=sub)
            model_aud, model_vis, model_av = Models
            model_zs = ZeroShotModel(eeg_dim=2600, shared_dim=256, num_classes=5)
            # if necessary edit num_layers
            model_eeg = MultiTaskShallowConvNet(Chans=30, Samples=500, num_layers=2)

            # please increase or decrease num_epochs for both models if necessary
            trainer = Trainer_eeg_multitask(model=model_eeg,
                                            data=[tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence,
                                                  te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence], num_epochs=500)
            trainer.train()
            out = zeroshot_training(Data=Data, Models=[model_aud, model_vis, model_eeg, model_av],
                                    ZeroShotModel=model_zs, epochs=100)
            results_zs.append(out)

            # Clear memory after each subject
            del model_zs, model_eeg, trainer, Data, Data_zs, Models
            clear_memory()
            clear_gpu_memory()

        except RuntimeError as e:
            print(f"Error processing subject {sub}: {e}")
            clear_memory()
            clear_gpu_memory()
            continue

        except Exception as e:
            print(f"Unexpected error processing subject {sub}: {e}")
            clear_memory()
            clear_gpu_memory()
            continue

    # results_zs will contain predicted labels for all models, and includes true labels as well
    # see Zeroshot_setting.zeroshot_training() return for more details
    #print("Predictions, can be used for further analysis: ", results_zs)
    # or save results_zs to file
    with open(results_path+f"\zeroshot_results_class{class_label}.pkl", "wb") as f:
        pickle.dump(results_zs, f)
    print(f"✅ Results saved to 'zeroshot_results_class{class_label}.pkl'")

