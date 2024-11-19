import sys
import numpy as np
import scipy.io.wavfile
import librosa

import noisereduce as nr
from scipy.signal import lfilter


def process_audio(in_path1, in_path2, out_path1, out_path2):
    # Flags to signal status to overlying handler
    # print("BUSY", file=sys.stdout)

    FS_TARGET = 16000

    # Read in audio
    fs, audio_data1 = scipy.io.wavfile.read(in_path1)
    fs, audio_data2 = scipy.io.wavfile.read(in_path2)

    # Normalize audio and convert to needed dtype
    audio_data1 = audio_data1.astype(np.float32)
    audio_data2 = audio_data2.astype(np.float32)

    audio_data1 = audio_data1 / np.max(np.abs(audio_data1))
    audio_data2 = audio_data2 / np.max(np.abs(audio_data2))

    # Resample if needed
    if fs != FS_TARGET:
        audio_data1 = librosa.resample(audio_data1, orig_sr=fs, target_sr=FS_TARGET)
        audio_data2 = librosa.resample(audio_data2, orig_sr=fs, target_sr=FS_TARGET)

    audio_data = np.column_stack((audio_data1, audio_data2))
    # processed_audio_data = audio_data
    
    window_size = 0.1  # Window size in seconds
    overlap_ratio = 0.5  # Overlap ratio between consecutive windows

    threshold = 0.5

    # Calculate window parameters
    window_length = int(window_size * fs)
    overlap_length = int(overlap_ratio * window_length)
    num_windows = int(
        np.floor((len(audio_data) - overlap_length) / (window_length - overlap_length))
    )

    # Initialize processed audio signal
    processed_audio_data = np.zeros_like(audio_data)

    # Loop through the audio signal
    for i in range(num_windows):
        # Extract current window
        start_index = (i - 1) * (window_length - overlap_length) + 1
        end_index = start_index + window_length
        window_data = audio_data[start_index:end_index, :]

        # Calculate energy of each channel within the window
        channel_energy = np.sum(window_data**2, axis=0) / window_length

        if abs(np.diff(channel_energy)) > threshold:

            # Find the channel with lower energy for this window
            channel_to_zero = np.argmin(channel_energy)

            # Zero out the channel with lower energy for this window
            window_data[:, channel_to_zero] = 0

            # Update the processed audio signal
            processed_audio_data[start_index:end_index, :] = window_data
        else:
            processed_audio_data[start_index:end_index, 0] = audio_data1[start_index:end_index]
            processed_audio_data[start_index:end_index, 1] = audio_data2[start_index:end_index]

    scipy.io.wavfile.write(out_path1, FS_TARGET, processed_audio_data[:, 0])
    scipy.io.wavfile.write(out_path2, FS_TARGET, processed_audio_data[:, 1])


def get_paths():
    return sys.stdin.readline().strip().split(",")


def main():
    while True:
        # Read input and output file paths from stdin
        input_path1, input_path2, output_path1, output_path2 = get_paths()
        process_audio(input_path1, input_path2, output_path1, output_path2)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        print("Processing audio directly from command line...")
        process_audio(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        print("Success, audio processed and saved to output files.")
    else:
        print("Starting main loop, waiting for input...")
        main()
