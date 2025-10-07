import torch
import os
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from transformers import AutoImageProcessor
from model.AMBT import AMBT
from model.AMBT_FACL import AMBT_FACL
from unimodal_models.Transformer_Video_concat import ViT_Encoder_Video
from unimodal_models.Transformer_Audio_concat import ViT_Encoder_Audio, ast_feature_extract
from unimodal_models.Transformer_EEG_concat import EEG_Encoder
import pickle

import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score


class VideoAudioDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rgb = self.inputs['rgb'][idx]
        spec = self.inputs['spectrogram'][idx]
        eeg = self.inputs['eeg'][idx]
        label = self.labels[idx]
        return {'rgb': rgb, 'spectrogram': spec, 'eeg': eeg}, label


def prepare_dataloader(x, y, batch_size, shuffle=False):
    dataset = VideoAudioDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


mod_path = r'C:\Users\user.DESKTOP-HI4HHBR\Downloads\facial_emotions_image_detection (1)'

for sub in range(1, 43):

    fusion_layer = 8
    contrastive_loss=True

    model_aud = ViT_Encoder_Audio(classifier=True, img_size=[1024, 128], in_chans=1, patch_size=(16, 16), stride=10,
                                  embed_pos=True, fusion_layer=fusion_layer)
    model_path = f"D:\.spyder-py3\Finetuned_models_ratio7030\model_with_weights_audio_finetuned_{sub}.pth"
    model_aud.load_state_dict(torch.load(model_path), strict=False)

    model_vid = ViT_Encoder_Video(classifier=True, img_size=(224, 224), in_chans=3, patch_size=(16, 16), stride=16,
                                  embed_pos=True, fusion_layer=fusion_layer)
    model_path = f"D:\.spyder-py3\Finetuned_models_ratio7030\model_with_weights_video_finetuned_{sub}.pth"
    model_vid.load_state_dict(torch.load(model_path), strict=False)

    num_layers = 4
    model_eeg = EEG_Encoder(nb_classes=5, Chans=30, Samples=500, num_layers=num_layers)
    path = os.path.join(r'D:\.spyder-py3\EEG_finetuned_models', f'subject_{sub:02d}_layers4_epochs400.pth')
    model_eeg.load_state_dict(torch.load(path), strict=False)

    file_name = f"subject_{sub:02d}_vis.pkl"
    file_ = os.path.join(r"D:\input images\Vision", file_name)

    with open(file_, 'rb') as f:
        vis_list2 = pickle.load(f)
    tr_x_vis, tr_y_vis, te_x_vis, te_y_vis = vis_list2
    data_video = [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis]
    processor = AutoImageProcessor.from_pretrained(mod_path)

    pixel_values_list = []
    for img_set in tr_x_vis:
        for img in img_set:
            processed = processor(images=img, return_tensors="pt")
            pixel_values = processed.pixel_values.squeeze()
            pixel_values_list.append(pixel_values)

    vals = torch.stack(pixel_values_list)

    pixel_values_list = []
    for img_set in te_x_vis:
        for img in img_set:
            processed = processor(images=img, return_tensors="pt")
            pixel_values = processed.pixel_values.squeeze()
            pixel_values_list.append(pixel_values)

    vals_test = torch.stack(pixel_values_list)

    direct = r"D:\input images\Audio"
    file_name = f"subject_{sub:02d}_aud.pkl"
    file_ = os.path.join(direct, file_name)

    with open(file_, 'rb') as f:
        vis_list2 = pickle.load(f)
    tr_x_vis, tr_y_vis, te_x_vis, te_y_vis = vis_list2
    tr_x_aud_ft = ast_feature_extract(tr_x_vis)
    te_x_aud_ft = ast_feature_extract(te_x_vis)
    tr_y_aud = tr_y_vis
    te_y_aud = te_y_vis

    file_name = f"subject_{sub:02d}_eeg.pkl"
    file_ = os.path.join(r"D:\input images\EEG", file_name)
    if os.path.exists(file_):
        with open(file_, 'rb') as f:
            eeg_list = pickle.load(f)
    else:
        print('Does not exist')

    tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = eeg_list
    tr_x_eeg = torch.from_numpy(tr_x_eeg).float().unsqueeze(1)  # Reshape to (batch, 1, chans, samples)
    te_x_eeg = torch.from_numpy(te_x_eeg).float().unsqueeze(1)  # Reshape to (batch, 1, chans, samples)
    data_eeg = [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg]

    data = [
        {'rgb': vals.view(-1, 25, 3, 224, 224),
         'spectrogram': torch.tensor(tr_x_aud_ft.unsqueeze(1), dtype=torch.float32),
         'eeg': tr_x_eeg},
        torch.from_numpy(tr_y_aud).long(),
        {'rgb': vals_test.view(-1, 25, 3, 224, 224),
         'spectrogram': torch.tensor(te_x_aud_ft.unsqueeze(1), dtype=torch.float32),
         'eeg': te_x_eeg},
        torch.from_numpy(te_y_aud).long(),
    ]

    tr_x, tr_y, te_x, te_y = data

    train_dataloader = prepare_dataloader(tr_x, tr_y, batch_size=2, shuffle=True)
    test_dataloader = prepare_dataloader(te_x, te_y, batch_size=2, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modalities = ['rgb', 'spectrogram', 'eeg']
    
    if contrastive_loss:
    
        ambt = AMBT_FACL(
            mlp_dim=3072, num_classes=5, num_layers=12,
            hidden_size=768, fusion_layer=fusion_layer, model_eeg=model_eeg,model_aud=model_aud,model_vid=model_vid
        )
    
    else:
       ambt = AMBT(
           mlp_dim=3072, num_classes=5, num_layers=12,
           hidden_size=768, fusion_layer=fusion_layer, model_eeg=model_eeg,model_aud=model_aud,model_vid=model_vid
       ) 

    criterion = torch.nn.CrossEntropyLoss()

    ambt = ambt.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        ambt = nn.DataParallel(ambt)

    modes = [True, False]

    for freeze in modes:

        for param in ambt.module.parameters():
            param.requires_grad = True

        for param in ambt.module.encoder.parameters():
            param.requires_grad = False
        for param in ambt.module.temporal_encoder_audio.parameters():
            param.requires_grad = False
        for param in ambt.module.temporal_encoder_rgb.parameters():
            param.requires_grad = False
        for param in ambt.module.model_eeg.parameters():
            param.requires_grad = False

        ambt.module.cls_token_aud.requires_grad = False
        ambt.module.cls_token_vid.requires_grad = False
        ambt.module.pos_embed_aud.requires_grad = False
        ambt.module.pos_embed_vid.requires_grad = False
        ambt.module.bottleneck.requires_grad = False
        if contrastive_loss:
            for param in ambt.module.lossav.parameters():
                param.requires_grad = True
            for param in ambt.module.lossev.parameters():
                param.requires_grad = True
            for param in ambt.module.lossae.parameters():
                param.requires_grad = True

        if freeze:
            epochs = 5

            optimizer = optim.Adam(ambt.parameters(), lr=5e-5)


        else:

            epochs = 20

            modalities1 = ['rgb', 'spectrogram']
            for modality in modalities1:
                for layer_idx in range(8, 12):  # Layers 8 to 12
                    layer = ambt.module.encoder.encoders[modality][layer_idx]
                    for param in layer.midsample.parameters():
                        param.requires_grad = True
                    for param in layer.endsample.parameters():
                        param.requires_grad = True

            for layer_idx in range(0, 4):  # Layers 8 to 12
                layer = ambt.module.encoder.encoders['eeg'][layer_idx]
                for param in layer.midsample.parameters():
                    param.requires_grad = True
                for param in layer.endsample.parameters():
                    param.requires_grad = True
            optimizer = optim.Adam(ambt.parameters(), lr=5e-5)

        for epoch in range(1, epochs + 1):
            ambt.train()
            train_correct, train_total = 0, 0
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(train_dataloader):
                optimizer.zero_grad()
                inputs['rgb'] = inputs['rgb'].view(-1, 3, 224, 224)

                inputs = {k: v.float().to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                
                if contrastive_loss:
                    outputs,loss_facl = ambt(inputs,labels)
                    if isinstance(outputs, dict):
                        ce_loss = []
                        for mod in outputs:
                            ce_loss.append(criterion(outputs[mod], labels))
                    else:
                        break         
                        
                    loss = torch.mean(torch.stack(ce_loss))+torch.mean(loss_facl)
                else:
                    outputs = ambt(inputs)
                    if isinstance(outputs, dict):
                        ce_loss = []
                        for mod in outputs:
                            ce_loss.append(criterion(outputs[mod], labels))
                    else:
                        break
    
                    loss = torch.mean(torch.stack(ce_loss))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 1 == 0:
                    # print(f'Epoch {epoch}, loss: {running_loss / 10:.4f}')
                    running_loss = 0

            ambt.eval()
            correct, total = 0, 0
            outputs_batch = []
            true_labels_batch = []
            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    inputs['rgb'] = inputs['rgb'].view(-1, 3, 224, 224)
                    inputs = {k: v.float().to(device) for k, v in inputs.items()}
                    labels = labels.to(device)
                    if contrastive_loss:
                        outputs = ambt(inputs,labels)
                    else:
                        outputs = ambt(inputs)

                    # Collect predictions and true labels for F1-score calculation
                    _, predicted = torch.max(outputs, dim=-1)
                    outputs_batch.extend(predicted.cpu().numpy())  # Add predictions
                    true_labels_batch.extend(labels.cpu().numpy())  # Add true labels

                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            test_accuracy = correct / total
            # Calculate F1-score
            f1 = f1_score(true_labels_batch, outputs_batch, average='weighted')

            print(f'Test accuracy after epoch {epoch}: {test_accuracy:.4f}')
            print(f'F1-score after epoch {epoch}: {f1:.4f}')

            with open('ambt_concat.txt', 'a') as f:
                f.write(
                    f'Subject {sub} Epoch {epoch} freeze:{freeze} Testing Accuracy: {test_accuracy:.4f} F1-score: {f1:.4f} \n')


