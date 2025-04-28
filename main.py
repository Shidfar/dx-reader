import os
import datetime
# from pathlib import Path
# from datasets.features.audio import Audio
import torch
# import sys
# import importlib
from langchain_community.document_loaders import PyPDFLoader
from bark import SAMPLE_RATE, generate_audio, preload_models
# from bark import api
from scipy.io.wavfile import write as write_wav
# from transformers import BarkModel, AutoProcessor
# import time
import numpy as np

import re

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
    # Create output directory for this PDF
    pdf_name = os.path.splitext(pdf_file)[0]
    pdf_voice_dir = os.path.join(voice_folder_path, pdf_name)
    os.makedirs(pdf_voice_dir, exist_ok=True)

    # Load PDF
    start = datetime.datetime.now()
    loader = PyPDFLoader(os.path.join(pdf_folder_path, pdf_file))
    doc_parts = loader.load()

    print(f"PDF has {len(doc_parts)} parts")

    voices = np.array(text_parts_to_speech(doc_parts))
    if len(voices) > 0:
        output_path = os.path.join(pdf_voice_dir, f"{pdf_name}.wav")
        write_wav(output_path, SAMPLE_RATE, voices)
        print(f"Completed {pdf_file}")
    else:
        print(f"Voices array is empty skipping file {pdf_file}")

    end = datetime.datetime.now()
    elapsed = end - start
    print(f"Elapsed time: {elapsed}")
    print("-----------------------------------")


def text_parts_to_speech(doc_parts, speaker="v2/en_speaker_2"):
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
                write_wav("data/voice/latest.wav", SAMPLE_RATE, np.array(voice))
                print("latest part written to data/voice/latest.wav")
                voices.extend(voice)
            else:
                print(f"  Part {part_idx + 1} is empty, skipping")

    return voices

def load_and_speak_pdfs(pdf_folder_path = './data/docs', voice_folder_path = './data/voice'):
    """
    Loads all PDFs from ./data/docs, converts their content to speech,
    and saves WAV files in ./data/voice/file_name/part_x.wav
    """
    # Ensure voice folder exists
    os.makedirs(voice_folder_path, exist_ok=True)

    # Get all PDF files
    doc_list = [s for s in os.listdir(pdf_folder_path) if s.endswith('.pdf')]
    num_of_docs = len(doc_list)

    if num_of_docs == 0:
        print("No PDF files found in ./data/docs")
        return

    general_start = datetime.datetime.now()
    print(f"Starting to process {num_of_docs} PDF documents...")

    for i, pdf_file in enumerate(doc_list):
        print(f"Processing {pdf_file} ({i + 1}/{num_of_docs})")
        file_to_speech(pdf_folder_path=pdf_folder_path, voice_folder_path=voice_folder_path, pdf_file=pdf_file)

    general_end = datetime.datetime.now()
    general_elapsed = general_end - general_start
    print(f"All PDFs processed in {general_elapsed}")
    print("-----------------------------------")


if __name__ == "__main__":
    print(f"torch MPS is available: {torch.backends.mps.is_available()}")
    preload_models()

    load_and_speak_pdfs()
