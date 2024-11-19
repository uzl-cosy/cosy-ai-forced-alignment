import sys
import numpy as np
import scipy.io.wavfile
import librosa
import torch
import warnings
import argparse
import os
from importlib.resources import files

# Global file descriptor variable, defaulting to None
FD = None


def send_pipe_message(message):
    global FD
    if FD is not None:
        os.write(FD, message.encode("utf-8") + b"\n")
        # os.fsync(FD)


def load_model():
    model = torch.jit.load(
        files("laboratorium_ai_forced_alignment.data").joinpath("Silero_Vad.pt")
    )
    send_pipe_message(
        "ready"
    )  # Signal that the model is loaded and ready only if FD is provided

    return model


def get_speech_timestamps(
    audio: torch.Tensor,
    model,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 1536,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
):
    """
    This method is used for splitting long audios into speech chunks using silero VAD
    Parameters
    ----------
    audio: torch.Tensor, one dimensional float torch.Tensor, other types are casted to torch if possible
    model: preloaded .jit silero VAD model
    threshold: float (default - 0.5)
        Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
        It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
    sampling_rate: int (default - 16000)
        Currently silero VAD models support 8000 and 16000 sample rates
    min_speech_duration_ms: int (default - 250 milliseconds)
        Final speech chunks shorter min_speech_duration_ms are thrown out
    min_silence_duration_ms: int (default - 100 milliseconds)
        In the end of each speech chunk wait for min_silence_duration_ms before separating it
    window_size_samples: int (default - 1536 samples)
        Audio chunks of window_size_samples size are fed to the silero VAD model.
        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate and 256, 512, 768 samples for 8000 sample rate.
        Values other than these may affect model perfomance!!
    speech_pad_ms: int (default - 30 milliseconds)
        Final speech chunks are padded by speech_pad_ms each side
    return_seconds: bool (default - False)
        whether return timestamps in seconds (default - samples)
    visualize_probs: bool (default - False)
        whether draw prob hist or not
    Returns
    ----------
    speeches: list of dicts
        list containing ends and beginnings of speech chunks (samples or seconds based on return_seconds)
    """

    if not torch.is_tensor(audio):
        try:
            audio = torch.Tensor(audio)
        except:
            raise TypeError("Audio cannot be casted to tensor. Cast it manually")

    if len(audio.shape) > 1:
        for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
            audio = audio.squeeze(0)
        if len(audio.shape) > 1:
            raise ValueError(
                "More than one dimension in audio. Are you trying to process audio with 2 channels?"
            )

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
        warnings.warn(
            "Sampling rate is a multiply of 16000, casting to 16000 manually!"
        )
    else:
        step = 1

    if sampling_rate == 8000 and window_size_samples > 768:
        warnings.warn(
            "window_size_samples is too big for 8000 sampling_rate! Better set window_size_samples to 256, 512 or 768 for 8000 sample rate!"
        )
    if window_size_samples not in [256, 512, 768, 1024, 1536]:
        warnings.warn(
            "Unusual window_size_samples! Supported window_size_samples:\n - [512, 1024, 1536] for 16000 sampling_rate\n - [256, 512, 768] for 8000 sampling_rate"
        )

    model.reset_states()
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000

    audio_length_samples = len(audio)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample : current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(
                chunk, (0, int(window_size_samples - len(chunk)))
            )
        speech_prob = model(chunk, sampling_rate).item()
        speech_probs.append(speech_prob)

    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15
    temp_end = 0

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech["start"] = window_size_samples * i
            continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech["end"] = temp_end
                if (
                    current_speech["end"] - current_speech["start"]
                ) > min_speech_samples:
                    speeches.append(current_speech)
                temp_end = 0
                current_speech = {}
                triggered = False
                continue

    if (
        current_speech
        and (audio_length_samples - current_speech["start"]) > min_speech_samples
    ):
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )

    if return_seconds:
        for speech_dict in speeches:
            speech_dict["start"] = round(speech_dict["start"] / sampling_rate, 1)
            speech_dict["end"] = round(speech_dict["end"] / sampling_rate, 1)
    elif step > 1:
        for speech_dict in speeches:
            speech_dict["start"] *= step
            speech_dict["end"] *= step

    return speeches


def process_audio(vad_model, in_path1, out_path1):
    # Flags to signal status to overlying handler
    # print("BUSY", file=sys.stdout)

    FS_TARGET = 16000

    # Read in audio
    fs, audio_data = scipy.io.wavfile.read(in_path1)

    # Normalize audio and convert to needed dtype
    audio_data = audio_data.astype(np.float32)

    audio_data = audio_data / (np.max(np.abs(audio_data)) + np.finfo(float).eps)

    # Resample if needed
    if fs != FS_TARGET:
        audio_data = librosa.resample(audio_data, orig_sr=fs, target_sr=FS_TARGET)

    # Initialize processed audio signal
    processed_audio_data = np.zeros_like(audio_data)

    thr = 0.8
    speech_timestamps = get_speech_timestamps(
        audio_data, vad_model, sampling_rate=FS_TARGET, threshold=thr
    )

    start_idx = []
    end_idx = []
    start_times = []
    end_times = []

    for current_timestamp in speech_timestamps:
        start_idx.append(current_timestamp["start"])
        end_idx.append(current_timestamp["end"])
        start_times.append(current_timestamp["start"] / 16000)
        end_times.append(current_timestamp["end"] / 16000)

    mask = np.ones(audio_data.shape, dtype=bool)
    for s_idx, e_idx in zip(start_idx, end_idx):
        mask[s_idx:e_idx] = False

    audio_data[mask] = 0

    scipy.io.wavfile.write(out_path1, FS_TARGET, audio_data)


def get_paths():
    return sys.stdin.readline().strip().split(",")


def main():
    parser = argparse.ArgumentParser(description="Process audio files with Whisper.")
    parser.add_argument(
        "-f",
        "--fd",
        type=int,
        help="Optional file descriptor for pipe communication",
    )
    args = parser.parse_args()

    if args.fd:
        global FD
        FD = args.fd  # Set the global file descriptor only if provided

    v_model = load_model()
    while True:
        # Read input and output file paths from stdin
        input_path1, output_path1 = get_paths()
        process_audio(v_model, input_path1, output_path1)


if __name__ == "__main__":
    # if len(sys.argv) == 3:
    #     print("Processing audio directly from command line...")
    #     print("Loading model...")
    #     v_model = load_model()
    #     process_audio(v_model, sys.argv[1], sys.argv[2])
    #     print("Success, audio processed and saved to output files.")
    # else:
    #     print("Starting main loop, waiting for input...")
    main()
