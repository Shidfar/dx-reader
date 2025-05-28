import datetime
import re
import os
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from bark import SAMPLE_RATE, generate_audio
from scipy.io.wavfile import write as write_wav


def slice_array_wave(input_array, amplitude_threshold, time_threshold, ignore_initial_steps=0):
    """
    Slice an input array based on consecutive low amplitude values.

    Parameters:
    - input_array (numpy.ndarray): Input array containing amplitude values.
    - amplitude_threshold (float): Threshold below which amplitudes are considered low.
    - time_threshold (int): Number of consecutive low amplitude values needed to trigger slicing.
    - ignore_initial_steps (int, optional): Number of initial steps to ignore before checking for consecutive low amplitudes.

    Returns:
    numpy.ndarray: Sliced array up to the point where consecutive low amplitudes reach the specified time_threshold.
    """

    low_amplitude_indices = np.abs(input_array) < amplitude_threshold

    consecutive_count = 0
    for i, is_low_amplitude in enumerate(low_amplitude_indices[ignore_initial_steps:]):
        if is_low_amplitude:
            consecutive_count += 1
        else:
            consecutive_count = 0

        if consecutive_count >= time_threshold:
            #             return input_array[:i - time_threshold]
            #             return input_array[:i - int(time_threshold/2) + int(time_threshold/4)]
            return input_array[:i + int(time_threshold / 4)]

    return input_array


def split_sentences(text):
    '''
    Given a text, return a list of sentences.
    '''
    ### Split on '. ', '.\n', '.\n\n', '!', '?', and ';'
    sentences = re.split(r'\. |\.\n|\.\n\n|!|\?|;', text)

    ### Strip whitespace from each sentence
    sentences = [
        sentence.strip() + '..'
        for sentence in sentences
    ]

    ### Remove empty strings from the list of sentences
    sentences = list(filter(None, sentences))

    number_of_sentences = len(sentences)

    return sentences[:-1], number_of_sentences - 1


def file_to_speech(pdf_folder_path, voice_folder_path, pdf_file):
    # Load PDF
    start = datetime.datetime.now()
    loader = PyPDFLoader(os.path.join(pdf_folder_path, pdf_file))
    doc_parts = loader.load()

    print(f"PDF has {len(doc_parts)} parts")

    voices = np.array(text_parts_to_speech(doc_parts))

    end = datetime.datetime.now()
    elapsed = end - start
    return voices, elapsed


def text_parts_to_speech(doc_parts, speaker="v2/en_speaker_2"):
    # AMPLITUDE_THRESHOLD = 0.05
    # TIME_THRESHOLD = int(SAMPLE_RATE * 0.7)  # sample_rate * n_seconds
    # IGNORE_INITIAL_STEPS = int(SAMPLE_RATE * 0.7)  # sample_rate * n_seconds

    voices = []
    # Process each part
    for part_idx, doc_part in enumerate(doc_parts):
        print(f"  Converting part {part_idx + 1}/{len(doc_parts)} to speech...")

        text_content = doc_part.page_content
        print(text_content)

        sentences, num = split_sentences(text_content)
        print(f"Reading {num} sentences in this chunk...")
        for i, sentence in enumerate(sentences):
            if sentence.strip():  # Only process non-empty content
                voice = generate_audio(sentence, history_prompt=speaker)
                #
                # sliced_voice = slice_array_wave(
                #     input_array=np.array(voice).squeeze(),
                #     amplitude_threshold=AMPLITUDE_THRESHOLD,
                #     time_threshold=TIME_THRESHOLD,
                #     ignore_initial_steps=IGNORE_INITIAL_STEPS
                # )
                #
                # write_wav("data/voice/latest_amplified.wav", SAMPLE_RATE, sliced_voice)
                write_wav("data/voice/latest.wav", SAMPLE_RATE, np.array(voice))
                print("latest part written to data/voice/latest.wav")
                # np.concatenate([voices, sliced_voice])
                voices.extend(voice)
            else:
                print(f"  Part {part_idx + 1} is empty, skipping")

    return voices
