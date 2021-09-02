import glob
import os
from params import *
import librosa
import numpy as np
import logging
import torch
import pescador

def get_files(folder_path, audio_extension):

    files = os.listdir(folder_path)
    output = []
    for f in files:
        if f.endswith(audio_extension):
            output.append(os.path.join(folder_path, f))
        else:
            print("file_broken or not exists!! : {}".format(os.path.join(folder_path, f)))
            sys.exit(-1)
    return output

def load_wav(wav_file_path):
    try:
        sampling_rate = librosa.get_samplerate(wav_file_path)
        audio_data, _ = librosa.load(wav_file_path, sr=sampling_rate)

        if normalize_audio:
            # Clip magnitude
            max_mag = np.max(np.abs(audio_data))
            if max_mag > 1:
                audio_data /= max_mag
    except Exception as e:
        print("Could not load {}: {}".format(wav_file_path, str(e)))
        raise e
    audio_len = len(audio_data)
    if audio_len < window_length:
        pad_length = window_length - audio_len
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad
        audio_data = np.pad(audio_data, (left_pad, right_pad), mode="constant")

    return audio_data.astype("float32")

def sample_audio(audio_data, start_idx=None, end_idx=None):
    audio_len = len(audio_data)
    if audio_len == window_length:
        # If we only have a single 1*window_length audio, just yield.
        sample = audio_data
    else:
        # Sample a random window from the audio
        if start_idx is None or end_idx is None:
            start_idx = np.random.randint(0, (audio_len - window_length) // 2)
            end_idx = start_idx + window_length
        sample = audio_data[start_idx:end_idx]
    sample = sample.astype("float32")
    assert not np.any(np.isnan(sample))
    return sample, start_idx, end_idx


def sample_buffer(buffer_data, start_idx=None, end_idx=None):
    audio_len = len(buffer_data) // 4
    if audio_len == window_length:
        # If we only have a single 1*window_length audio, just yield.
        sample = buffer_data
    else:
        # Sample a random window from the audio
        if start_idx is None or end_idx is None:
            start_idx = np.random.randint(0, (audio_len - window_length) // 2)
            end_idx = start_idx + window_length
        sample = buffer_data[start_idx * 4 : end_idx * 4]
    return sample, start_idx, end_idx



def wav_generator(file_path):
    audio_data = load_wav(file_path)
    while True:
        sample, _, _ = sample_audio(audio_data)
        yield {"single": sample}

def create_stream_reader(files):
    data_streams = []
    for f in files:
        stream = pescador.Streamer(wav_generator, f)
        data_streams.append(stream)
    mux = pescador.ShuffledMux(data_streams)
    batch_gen = pescador.buffer_stream(mux, batch_size)
    return batch_gen

class WaveLoader:
    def __init__(self, folder_path, audio_extension):
        self.files = get_files(folder_path, audio_extension)
        self.data_iter = None
        self.initialize_iterator()

    def initialize_iterator(self):
        data_iter = create_stream_reader(self.files)
        self.data_iter = iter(data_iter)
    
    def __len__(self):
        return len(self.files)

    
    def numpy_to_tensor(self, numpy_array):
        numpy_array = numpy_array[:, np.newaxis, :]
        return torch.Tensor(numpy_array).to(device)

    def __iter__(self):
        return self

    def __next__(self):
        x = next(self.data_iter)
        return self.numpy_to_tensor(x["single"])