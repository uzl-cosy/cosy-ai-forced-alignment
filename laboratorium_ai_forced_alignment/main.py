import argparse
from importlib.resources import files
import sys
import json
import re
import os
import numpy as np
import torch
import torchaudio
import librosa
import scipy.io.wavfile
import string
import nltk
from nltk.tokenize import sent_tokenize

from num2words import num2words

# Global file descriptor variable, defaulting to None
FD = None


def send_pipe_message(message):
    global FD
    if FD is not None:
        os.write(FD, message.encode("utf-8") + b"\n")
        # os.fsync(FD)


def load_model():
    nltk.download("punkt_tab")
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    send_pipe_message(
        "ready"
    )  # Signal that the model is loaded and ready only if FD is provided

    return model, tokenizer, aligner


def create_empty_output():
    sentences = []
    sentence_start_times = []
    sentence_end_times = []
    words = []
    start_times_alignment = []
    end_times_alignment = []
    confidences = []

    empty_output = {
        "Sentences": sentences,
        "Start Times": sentence_start_times,
        "End Times": sentence_end_times,
        "Words": words,
        "Word Start Times": start_times_alignment,
        "Word End Times": end_times_alignment,
        "Confidences": confidences,
    }
    return empty_output


def get_word_times(waveform_length, token_in_span, num_frames, sample_rate=16000):
    ratio = waveform_length / num_frames
    x0 = int(ratio * token_in_span[0].start)
    x1 = int(ratio * token_in_span[-1].end)
    return x0 / sample_rate, x1 / sample_rate


def detect_number_index(strings):
    pattern = r"\d+"  # Matches one or more digits
    numbers_with_index = []
    for i, string in enumerate(strings):
        numbers = re.findall(pattern, string)
        if numbers:  # If numbers are found in the string
            numbers_with_index.append(i)
    return numbers_with_index


def normalize_uroman(text):

    text = text.lower()  # Convert String to lowercase
    text = text.replace("’", "'")
    text = re.sub(
        r"(\d+)", lambda x: num2words(int(x.group(0)), lang="de"), text
    )  # NEW!!! Nummer -->  Wort
    text = (
        text.replace("ö", "oe")
        .replace("ä", "ae")
        .replace("ü", "ue")
        .replace("Ö", "Oe")
        .replace("Ä", "Ae")
        .replace("Ü", "Ue")
        .replace("ß", "ss")
    )  # Umlaute ersetzen
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip()


def tokenize_sentences_with_spans(text):
    sentences = sent_tokenize(text, language="german")
    spans = []
    start = 0
    for sentence in sentences:
        start_index = text.find(sentence, start)
        end_index = start_index + len(sentence)
        spans.append((start_index, end_index))
        start = end_index
    return sentences, spans


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound


def read_io_paths():
    return sys.stdin.readline().strip().split(",")


def process(
    a_model,
    a_tokenizer,
    a_aligner,
    input_path_json,
    input_path_audio,
    output_path_json,
):

    # Transcribe the audio file
    RATE = 16000

    fs, audio_data = scipy.io.wavfile.read(input_path_audio)

    if audio_data.size == 0:
        return sys.stdout.write("Audio Data empty!")

    audio_data = int2float(audio_data)
    audio_data = audio_data / (np.max(np.abs(audio_data)) + np.finfo(float).eps)

    if fs != RATE:
        audio_data = librosa.resample(audio_data, orig_sr=fs, target_sr=RATE)

    with open(input_path_json, "r") as f:
        data = json.load(f)

    with open(
        files("laboratorium_ai_forced_alignment.data").joinpath(
            "WhisperFilterList.json"
        ),
        "r",
    ) as f:
        filter_list = json.load(f)
    filter_list = filter_list["Hallucination Sentences"]

    if "Text" not in data.keys():
        sys.stdout.write(
            "No transcription to align. Create empty output file and wait new data to process!"
        )

        out_dict = create_empty_output()

        with open(output_path_json, "w") as f:
            json.dump(out_dict, f, ensure_ascii=False)

        send_pipe_message("done")

    elif not data["Text"].strip():
        sys.stdout.write(
            "No transcription to align. Create empty output file and wait new data to process!"
        )

        out_dict = create_empty_output()

        with open(output_path_json, "w") as f:
            json.dump(out_dict, f, ensure_ascii=False)

        send_pipe_message("done")
    else:
        transcript = data["Text"]
        transcript = transcript.replace("'", "")
        transcript = transcript.replace("...", "")  # NEU --> Führt zu Fehler?
        sentences, sentence_spans = tokenize_sentences_with_spans(transcript)

        word_tokens = nltk.tokenize.word_tokenize(
            transcript, language="german", preserve_line=False
        )

        in_text = normalize_uroman(transcript).split()

        words = []
        word_spans = []
        start = 0

        for token in word_tokens:
            start = transcript.find(token, start)
            end = start + len(token)
            # word_spans.append((start, end))
            if token not in string.punctuation and not token == "...":
                words.append(token)
                word_spans.append((start, end))

        with torch.inference_mode():
            emission, _ = a_model(torch.tensor(audio_data).unsqueeze(0))
            token_spans = a_aligner(emission[0], a_tokenizer(in_text))

        start_times_alignment = []
        end_times_alignment = []

        for idx in range(len(in_text)):
            start_time, end_time = get_word_times(
                torch.tensor(audio_data).unsqueeze(0).size(1),
                token_spans[idx],
                emission.size(1),
            )
            start_times_alignment.append(start_time)
            end_times_alignment.append(end_time)

        # start_times = start_times_alignment.copy()
        # end_times = end_times_alignment.copy()

        index_hyphens = []
        for h_idx in range(len(words)):
            if "-" in words[h_idx]:
                h_counter = 0
                for c in words[h_idx]:
                    if c == "-":
                        h_counter += 1
                index_hyphens.append((h_idx, h_counter))

        confidences = []
        for token_span in token_spans:
            score = 0
            for token_data in token_span:
                score += token_data.score
            confidences.append(np.round(score / len(token_span), 4))

        for del_index in index_hyphens:
            confidences[del_index[0]] = np.round(
                np.mean(confidences[del_index[0] : del_index[0] + del_index[1] + 1]),
                decimals=4,
            )
            for _ in range(del_index[1]):
                del start_times_alignment[del_index[0] + 1]
                del end_times_alignment[del_index[0]]
                del confidences[del_index[0] + 1]

        sentence_start_times = []
        sentence_end_times = []
        for sentence_span in sentence_spans:
            s_idx = [
                index
                for index, tup in enumerate(word_spans)
                if tup[0] == sentence_span[0]
            ][0]

            # e_idx = [index for index, tup in enumerate(word_spans) if tup[1] == sentence_span[1] - 1][-1]
            # sentence_start_times.append(start_times_alignment[s_idx])
            # sentence_end_times.append(end_times_alignment[e_idx])

            try:
                e_idx = [
                    index
                    for index, tup in enumerate(word_spans)
                    if tup[1] == sentence_span[1] - 1
                ][-1]
            except:
                e_idx = len(end_times_alignment) - 1

            sentence_start_times.append(start_times_alignment[s_idx])

            try:
                sentence_end_times.append(end_times_alignment[e_idx])
            except:
                sentence_end_times.append(end_times_alignment[-1])

        # Confidence Filter
        confidences_filter_threshold = 0.1
        sentence_lens = []
        m_res = []
        c_filter_counters = []
        c_filter_counter = 0
        s_idx_confidences = 0
        e_idx_confidences = 0
        for sentence in sentences:
            e_idx_confidences += len(sentence.split())
            sentence_lens.append(len(sentence.split()))
            mean_val_sentence = np.round(
                np.mean(confidences[s_idx_confidences:e_idx_confidences]), 5
            )
            m_res.append(mean_val_sentence)

            if (
                mean_val_sentence < confidences_filter_threshold
                and re.sub(r"\.$", "", sentence) in filter_list
            ):
                c_filter_counters.append(c_filter_counter)

            c_filter_counter += 1
            s_idx_confidences += e_idx_confidences

        for c_filter_cnt in reversed(c_filter_counters):
            num_words_till_sentence_start = sum(sentence_lens[:c_filter_cnt])
            sentence_length = sentence_lens[c_filter_cnt]

            del words[
                num_words_till_sentence_start : num_words_till_sentence_start
                + sentence_length
            ]
            del start_times_alignment[
                num_words_till_sentence_start : num_words_till_sentence_start
                + sentence_length
            ]
            del end_times_alignment[
                num_words_till_sentence_start : num_words_till_sentence_start
                + sentence_length
            ]
            del confidences[
                num_words_till_sentence_start : num_words_till_sentence_start
                + sentence_length
            ]
            del (
                sentences[c_filter_cnt],
                sentence_start_times[c_filter_cnt],
                sentence_end_times[c_filter_cnt],
            )

        out_dict = {
            "Sentences": sentences,
            "Start Times": sentence_start_times,
            "End Times": sentence_end_times,
            "Words": words,
            "Word Start Times": start_times_alignment,
            "Word End Times": end_times_alignment,
            "Confidences": confidences,
            # "Debug Start Times": start_times,
            # "Debug End Times": end_times
        }

        with open(output_path_json, "w") as f:
            json.dump(out_dict, f, ensure_ascii=False)

        send_pipe_message("done")


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

    align_asr_model, align_tokenizer, align_aligner = load_model()
    while True:
        # Read input and output file paths from stdin
        input_path_json, input_path_audio, output_path_json = read_io_paths()
        process(
            align_asr_model,
            align_tokenizer,
            align_aligner,
            input_path_json,
            input_path_audio,
            output_path_json,
        )


if __name__ == "__main__":
    main()
