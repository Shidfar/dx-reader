from coqui.synthesizer import Synthesizer
from coqui.api import TTS
from coqui.tts.utils.manage import ModelManager
from pathlib import Path
import torch
import sys
import os
import importlib


def speak(text):
    # Fix import paths - create an alias from coqui.tts to TTS
    sys.modules['TTS'] = importlib.import_module('coqui.tts')

    path = Path(__file__).parent / "../.models.json"
    manager = ModelManager(path, progress_bar=True)
    # api = TTS()

    vocoder_name = None
    vocoder_path = None
    vocoder_config_path = None
    vc_path = None
    vc_config_path = None
    model_dir = None
    speakers_file_path = None
    language_ids_file_path = None

    # Check if CUDA is available and set device accordingly
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"Using device: {device}")

    # model_path, config_path, model_item = manager.download_model("tts_models/en/ljspeech/glow-tts")
    model_path, config_path, model_item = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")

    # tts model
    tts_path = None
    tts_config_path = None
    if model_item["model_type"] == "tts_models":
        tts_path = model_path
        tts_config_path = config_path
        if "default_vocoder" in model_item:
            vocoder_name = model_item["default_vocoder"]

    # voice conversion model
    if model_item["model_type"] == "voice_conversion_models":
        vc_path = model_path
        vc_config_path = config_path

    # tts model with multiple files to be loaded from the directory path
    if model_item.get("author", None) == "fairseq" or isinstance(model_item["model_url"], list):
        model_dir = model_path
        tts_path = None
        tts_config_path = None
        vocoder_name = None

    # Try loading vocoder with error handling
    if vocoder_name is not None and not vocoder_path:
        try:
            vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)
            print(f"Successfully loaded vocoder: {vocoder_name}")
        except Exception as e:
            print(f"Error loading vocoder {vocoder_name}: {str(e)}")
            print("Continuing without a vocoder (will use Griffin-Lim algorithm instead)")
            vocoder_path = None
            vocoder_config_path = None

    # CASE4: set custom model paths
    if model_path is not None:
        tts_path = model_path
        tts_config_path = config_path

    try:
        # Initialize synthesizer with proper parameter names
        synthesizer = Synthesizer(
            tts_checkpoint=tts_path,
            tts_config_path=tts_config_path,
            tts_speakers_file=speakers_file_path,
            tts_languages_file=language_ids_file_path,
            vocoder_checkpoint=vocoder_path,
            vocoder_config=vocoder_config_path,
            encoder_checkpoint=None,
            encoder_config=None,
            vc_checkpoint=vc_path,
            vc_config=vc_config_path,
            model_dir=model_dir,
            voice_dir=None,
            use_cuda=use_cuda  # Use the detected CUDA availability
        )


        print(" > Text: {}".format(text))

        # kick it
        if tts_path is not None:
            wav = synthesizer.tts(
                text=text,
                speaker_name=None,  # speaker_idx,
                language_name=None,  # args.language_idx,
                speaker_wav=None,  # args.speaker_wav,
                reference_wav=None,  # args.reference_wav,
                style_wav=None,  # args.capacitron_style_wav,
                style_text=None,  # args.capacitron_style_text,
                reference_speaker_name=None,  # args.reference_speaker_idx,
            )
        elif vc_path is not None:
            wav = synthesizer.voice_conversion(
                source_wav=None,  # args.source_wav,
                target_wav=None,  # args.target_wav,
            )
        elif model_dir is not None:
            wav = synthesizer.tts(
                text=text,
                speaker_name=None,  # args.speaker_idx,
                language_name=None,  # args.language_idx,
                speaker_wav=None,  # args.speaker_wav
            )

        # save the results
        output = "output.wav"
        print(" > Saving output to {}".format(output))
        synthesizer.save_wav(wav, output)

    except Exception as e:
        print(f"Error: {str(e)}")

        # Let's try a different approach - use the TTS API directly
        print("\nFalling back to using the TTS API directly...")

        model_name = "tts_models/en/ljspeech/tacotron2-DDC"
        print(f"Loading model {model_name}...")

        tts = TTS(model_name=model_name)

        print(" > Text: {}".format(text))

        output = "output.wav"
        print(" > Saving output to {}".format(output))
        tts.tts_to_file(text=text, file_path=output)
