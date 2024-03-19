import os
from glob import glob

import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from torchaudio.sox_effects import apply_effects_tensor

class Transform(torch.nn.Module):
    def __init__(self, sample_rate, trigger_level=7.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.trigger_level = trigger_level
        self.vad = T.Vad(sample_rate=self.sample_rate, trigger_level=self.trigger_level)

    def forward(self, waveform, native_sample_rate):
        # Voice Activity Detection
        waveform = torch.flip(waveform, [-1])
        waveform = self.vad(waveform)
        waveform = torch.flip(waveform, [-1])
        waveform = self.vad(waveform)

        # Resample
        waveform = F.resample(waveform, native_sample_rate, self.sample_rate)

        # Convert to mono
        if len(waveform.size()) == 1: # don't have explicit channel dimension, and mono
            waveform = waveform.unsqueeze(0)
        elif len(waveform.size()) == 2: # don't have explicit channel dimension
            if waveform.size(0) == 1:
                pass
            else:
                waveform = waveform.mean(0, keepdim=True)
        elif len(waveform.size()) == 3: # have explicit channel dimension
            if waveform.size(1) == 1:
                waveform = waveform.squeeze(1)
            else:
                waveform = waveform.mean(1, keepdim=False)

        return waveform

def parse_speaker_df(file_path):
    df = pd.read_csv(file_path, sep='|', skiprows=12, names = ['ID', 'SEX', 'SUBSET', 'MINUTES', 'NAME'])
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    return df

def main(args):
    dataset_dir = args.dataset_dir
    sample_rate = args.sample_rate
    duration = args.duration
    target_dir = os.path.join(os.path.dirname(dataset_dir), f'LibriSpeech_{int(sample_rate//1000)}kHz_{duration}s')
    
    # Load the speaker information
    speaker_df = parse_speaker_df(os.path.join(dataset_dir, 'SPEAKERS.TXT'))
    speaker_df = speaker_df[speaker_df['SUBSET']=='dev-clean']
    speaker_id = speaker_df.ID.to_list()
    speaker_sex = speaker_df.SEX.to_list()

    # Split the dataset into training and testing sets, using stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(speaker_id, speaker_sex)):
        if fold == 0:
            train_speaker_id = [str(speaker_id[i]) for i in train_index]
            test_speaker_id = [str(speaker_id[i]) for i in test_index]
            break

    # Create a target directory
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get a list of all the audio files
    audio_files = glob(f'{dataset_dir}/**/*.flac', recursive=True)

    # Create a transform to convert the audio to mono
    transform = Transform(sample_rate)

    # Apply the transforms to the audio files
    for i, audio_file in tqdm(enumerate(audio_files), total=len(audio_files), desc='Processing'):
        # Get speaker ID and basename
        basename = os.path.splitext(os.path.basename(audio_file))[0]
        speaker_id_temp = audio_file.split('/')[-3]
        is_train = speaker_id_temp in train_speaker_id

        # Load the audio
        waveform, native_sample_rate = torchaudio.load(audio_file)

        # VAD, resample, and convert to mono
        waveform = transform(waveform, native_sample_rate)

        # Windowing
        if waveform.size(-1) < int(sample_rate * duration):
            continue
        waveform = waveform.unfold(-1, int(sample_rate * duration), int(sample_rate * duration))

        # Save the processed audio
        for j in range(waveform.size(1)):
            target_file = os.path.join(target_dir, ['val', 'train'][is_train], f'{basename}_{j}.wav')
            target_dir_temp = os.path.dirname(target_file)

            if not os.path.exists(target_dir_temp):
                os.makedirs(target_dir_temp)
                
            torchaudio.save(target_file, waveform[:,j,:], sample_rate)

    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concentration bounds')
    parser.add_argument('-dset', '--dataset_dir', help='a directory for LibriSpeech dataset', default='../data/LibriSpeech')
    parser.add_argument('-sr', '--sample_rate', help='a target sampling rate', default=16000, type = int)
    parser.add_argument('-d', '--duration', help='a target duration', default=1, type = int)
    args = parser.parse_args()

    main(args)