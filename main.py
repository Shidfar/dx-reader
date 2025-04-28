import os
import datetime
import torch
from speak import file_to_speech
from bark import SAMPLE_RATE
from bark import preload_models
from scipy.io.wavfile import write as write_wav


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
        # Create output directory for this PDF
        pdf_name = os.path.splitext(pdf_file)[0]
        pdf_voice_dir = os.path.join(voice_folder_path, pdf_name)
        os.makedirs(pdf_voice_dir, exist_ok=True)

        voices, elapsed = file_to_speech(pdf_folder_path=pdf_folder_path, voice_folder_path=voice_folder_path, pdf_file=pdf_file)

        if len(voices) > 0:
            output_path = os.path.join(pdf_voice_dir, f"{pdf_name}.wav")
            write_wav(output_path, SAMPLE_RATE, voices)
            print(f"Completed {pdf_file}")
        else:
            print(f"Voices array is empty skipping file {pdf_file}")

        print(f"Elapsed time: {elapsed}")
        print("-----------------------------------")

    general_end = datetime.datetime.now()
    general_elapsed = general_end - general_start
    print(f"All PDFs processed in {general_elapsed}")
    print("-----------------------------------")


if __name__ == "__main__":
    print(f"torch MPS is available: {torch.backends.mps.is_available()}")
    preload_models()
    load_and_speak_pdfs()
