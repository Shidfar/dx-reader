import os
import datetime
from pathlib import Path
import torch
import sys
import importlib
from langchain_community.document_loaders import PyPDFLoader
from coqui.synthesizer import Synthesizer
from coqui.api import TTS
from coqui.tts.utils.manage import ModelManager



import torch.serialization
# --- Other imports like os, datetime, Path, sys, importlib ---

try:
    # Adjust this import path if your TTS installation structure is different
    from coqui.tts.tts.configs.xtts_config import XttsConfig

    # Allowlist the XttsConfig class globally for torch.load
    # This needs to run BEFORE the code that triggers the internal torch.load
    torch.serialization.add_safe_globals([XttsConfig])
    print("Successfully added XttsConfig to torch safe globals list.")

except ImportError:
    print("Warning: Could not import XttsConfig from TTS library.")
    print("         Make sure Coqui TTS is installed correctly.")
    print("         The torch.load error might persist if the class cannot be allowlisted.")
except AttributeError:
    print("Warning: torch.serialization.add_safe_globals not found.")
    print("         Ensure you are using a compatible PyTorch version (>= 2.6 ideally).")



def load_and_speak_pdfs():
    """
    Loads all PDFs from ./data/docs, converts their content to speech,
    and saves WAV files in ./data/voice/file_name/part_x.wav
    """
    pdf_folder_path = './data/docs'
    voice_folder_path = './data/voice'

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

    # Initialize speech synthesizer once to reuse
    synthesizer = initialize_synthesizer()

    # for i, pdf_file in enumerate(doc_list):
    for i in range(0, 1):
        pdf_file = doc_list[0]
        print(f"Processing {pdf_file} ({i+1}/{num_of_docs})")

        # Create output directory for this PDF
        pdf_name = os.path.splitext(pdf_file)[0]
        pdf_voice_dir = os.path.join(voice_folder_path, pdf_name)
        os.makedirs(pdf_voice_dir, exist_ok=True)

        # Load PDF
        start = datetime.datetime.now()
        loader = PyPDFLoader(os.path.join(pdf_folder_path, pdf_file))
        doc_parts = loader.load()

        print(f"PDF has {len(doc_parts)} parts")

        # Process each part
        # for part_idx, doc_part in enumerate(doc_parts):
        for part_idx in range(0, 1):
            print(f"  Converting part {part_idx+1}/{len(doc_parts)} to speech...")

            text_content = doc_parts[0].page_content
            print(text_content)

            if text_content.strip():  # Only process non-empty content
                output_path = os.path.join(pdf_voice_dir, f"testing-part_{part_idx+1}.wav")
                speak_text(text_content, output_path, synthesizer)
                print(f"  Saved to {output_path}")
            else:
                print(f"  Part {part_idx+1} is empty, skipping")

        end = datetime.datetime.now()
        elapsed = end - start
        print(f"Completed {pdf_file} in {elapsed}")
        print("-----------------------------------")

    general_end = datetime.datetime.now()
    general_elapsed = general_end - general_start
    print(f"All PDFs processed in {general_elapsed}")
    print("-----------------------------------")


def initialize_synthesizer():
    """Initialize and return a TTS synthesizer"""
    # Fix import paths - create an alias from coqui.tts to TTS
    sys.modules['TTS'] = importlib.import_module('coqui.tts')

    path = Path(__file__).parent / ".models.json"
    manager = ModelManager(path, progress_bar=True)

    # Check if CUDA is available and set device accordingly
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"Using device: {device}")

    try:
        # First try the XTTS v2 model
        model_path, config_path, model_item = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")

        # Initialize synthesizer parameters
        tts_path = model_path
        tts_config_path = config_path
        vocoder_path = None
        vocoder_config_path = None

        # Try to get vocoder if available
        if "default_vocoder" in model_item:
            try:
                vocoder_path, vocoder_config_path, _ = manager.download_model(model_item["default_vocoder"])
            except Exception as e:
                print(f"Error loading vocoder: {e}")
                print("Continuing without a vocoder")

        # Initialize the synthesizer
        synthesizer = Synthesizer(
            tts_checkpoint=tts_path,
            tts_config_path=tts_config_path,
            # tts_speakers_file=None,
            # tts_languages_file=None,
            vocoder_checkpoint=vocoder_path,
            vocoder_config=vocoder_config_path,
            # encoder_checkpoint=None,
            # encoder_config=None,
            # vc_checkpoint=None,
            # vc_config=None,
            # model_dir=None,
            voice_dir=None,
            use_cuda=use_cuda
        )

        return synthesizer

    except Exception as e:
        print(f"Error initializing synthesizer: {e}")
        print("Falling back to TTS API")
        return None


def speak_text(text, output_path, synthesizer=None):
    """
    Convert text to speech and save to the specified output path

    Args:
        text (str): Text to convert to speech
        output_path (str): Path to save the WAV file
        synthesizer: Optional pre-initialized synthesizer
    """
    try:
        if synthesizer:
            # Use the existing synthesizer
            wav = synthesizer.tts(
                text=text,
                speaker_name=None,
                language_name=None,
                speaker_wav=None,
                reference_wav=None,
                style_wav=None,
                style_text=None,
                reference_speaker_name=None,
            )
            synthesizer.save_wav(wav, output_path)
        else:
            # Fallback to direct API
            # tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
            tts.tts_to_file(text=text, file_path=output_path)

    except Exception as e:
        print(f"Error generating speech: {e}")
        # print("Falling back to simpler TTS model")
        # try:
        #     tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        #     tts.tts_to_file(text=text, file_path=output_path)
        # except Exception as e2:
        #     print(f"Failed to generate speech: {str(e2)}")


if __name__ == "__main__":
    synthesizer = initialize_synthesizer()
    str = """
    
Okay, that strategy – using the slower charger at home (where it's often plugged in) and the faster charger when mobile – is perfectly reasonable and safe. Let's break down the implications:

**Using the Slower Charger at Home:**

1.  **Adequate for Light Use:** If you're primarily doing light tasks (Browse, email, documents) at home while plugged in, the slower charger will likely be sufficient to keep the battery topped up or charge it slowly.
2.  **Potential Power Shortfall:** If you perform demanding tasks at home (video editing, compiling, gaming), the slower charger might not provide enough power. You could see the battery drain slowly even while plugged in. This would be the main drawback – ensuring it actually meets your power needs *during* use at home.
3.  **Battery Health (Plugged In):** Whether you use the slow or fast charger, if the MacBook is plugged in for extended periods, macOS's Battery Health Management will engage. It will manage the charge level (often holding it around 80%) to minimize stress, regardless of the charger's wattage. So, using the slower charger here doesn't offer a significant *additional* battery health advantage over the fast charger in this specific "plugged-in" scenario, as the OS is already optimizing.
4.  **Convenience:** You might appreciate having a dedicated charger that stays at your desk.

**Using the Faster Charger On The Go:**

1.  **Faster Top-Ups:** This is ideal for mobile use where you might only have limited time to plug in and need to recharge quickly.
2.  **Sufficient Power:** Ensures you have enough power even if you're doing demanding work while charging on the go.


    """.strip()
    speak_text(str, "./data/test/test.wav", synthesizer)
    # load_and_speak_pdfs()
