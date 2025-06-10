import os
import torchaudio
from torchaudio.transforms import Resample
from transformers import ASTFeatureExtractor
import numpy as np
import pickle
from EAV_datasplit import *
import Transformer_Audio

class DataLoadAudio:
    def __init__(self, subject='all', parent_directory=r'D:\Dropbox\DATASETS\EAV', target_sampling_rate=16000):
        self.parent_directory = parent_directory
        self.original_sampling_rate = int()
        self.target_sampling_rate = target_sampling_rate
        self.subject = subject
        self.file_path = list()
        self.file_emotion = list()

        self.seg_length = 5  # 5s
        self.feature = None
        self.label = None
        self.label_indexes = None
        self.test_prediction = list()

    def data_files(self):
        subject = f'subject{self.subject:02d}'
        file_emotion = []
        subjects = []
        path = os.path.join(self.parent_directory, subject, 'Audio')
        for i in os.listdir(path):
            emotion = i.split('_')[4]
            self.file_emotion.append(emotion)
            self.file_path.append(os.path.join(path, i))

    def feature_extraction(self):
        x = []
        y = []
        feature_extractor = ASTFeatureExtractor()
        for idx, path in enumerate(self.file_path):
            waveform, sampling_rate = torchaudio.load(path)
            self.original_sampling_rate = sampling_rate
            if self.original_sampling_rate is not self.target_sampling_rate:
                resampler = Resample(orig_freq=sampling_rate, new_freq=self.target_sampling_rate)
                resampled_waveform = resampler(waveform)
                resampled_waveform = resampled_waveform.squeeze().numpy()
            else:
                resampled_waveform = waveform

            segment_length = self.target_sampling_rate * self.seg_length
            num_sections = int(np.floor(len(resampled_waveform) / segment_length))

            for i in range(num_sections):
                t = resampled_waveform[i * segment_length: (i + 1) * segment_length]
                x.append(t)
                y.append(self.file_emotion[idx])
        print(f"Original sf: {self.original_sampling_rate}, resampled into {self.target_sampling_rate}")

        emotion_to_index = {
            'Neutral': 0,
            'Happiness': 3,
            'Sadness': 1,
            'Anger': 2,
            'Calmness': 4
        }
        y_idx = [emotion_to_index[emotion] for emotion in y]
        self.feature = np.squeeze(np.array(x))
        self.label_indexes = np.array(y_idx)
        self.label = np.array(y)

    def process(self):
        self.data_files()
        self.feature_extraction()
        return self.feature, self.label_indexes

    def label_emotion(self):
        self.data_files()
        self.feature_extraction()
        return self.label

if __name__ == "__main__":
    test_acc = []
    for sub_idx in range(1,43):
        aud_loader = DataLoadAudio(subject=sub_idx, parent_directory=r'D:\Dropbox\DATASETS\EAV')
        [data_aud , data_aud_y] = aud_loader.process()
        division_aud = EAVDataSplit(data_aud, data_aud_y)
        [tr_x_aud, tr_y_aud, te_x_aud , te_y_aud] = division_eeg.get_split(h_idx=56)
        data = [tr_x_aud, tr_y_aud, te_x_aud , te_y_aud]
        file_path = "D:/Dropbox/Datasets/EAV/Input_images/Audio/"
        file_name = f"subject_{sub_idx:02d}_aud.pkl"
        file_ = os.path.join(file_path, file_name)
        Aud_list = [tr_x_aud, tr_y_aud, te_x_aud, te_y_aud]

        with open(file_, 'wb') as f:
            pickle.dump(Aud_list, f)