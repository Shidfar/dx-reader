import torch
from scipy.io.wavfile import write as write_wav
from transformers import AutoProcessor, BarkModel, AutoModel

from bark import SAMPLE_RATE, generate_audio, preload_models
import time
import os

import warnings

warnings.filterwarnings('ignore')

import gc
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io.wavfile import write  # to save audios

### torch
from torchaudio.transforms import Resample
import torchaudio

### audio
import transformers

from torch.nn import Parameter

def choose_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for acceleration")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) for acceleration")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")
    return device

device = choose_device()

class CFG:
    DEVICE = device

    ### processor
    SPEAKER = "v2/en_speaker_0"  # voice preset

    ### model
    MODEL_NAME = 'suno/bark'

    ### post-processing: to visualize and remove Noise
    AMPLITUDE_THRESHOLD = 0.05
    TIME_THRESHOLD = int(24_000 * 0.5)  # sample_rate * n_seconds
    IGNORE_INITIAL_STEPS = int(24_000 * 0.5)  # sample_rate * n_seconds


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) + \
        sum(p.numel() for p in model.parameters() if not p.requires_grad and isinstance(p, Parameter))


def count_tokens(text, processor):
    return len(processor.tokenizer(text)['input_ids'])


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


def spike():
    print('transformers version: ', transformers.__version__)
    print('torch version: ', torch.__version__)

    # Check CUDA availability and warn if not available
    # if not torch.cuda.is_available():
    #     print("WARNING: CUDA is not available. Running on CPU will be slow.")
    #     USE_CUDA = False
    # else:
    #     USE_CUDA = True

    text_to_infer = '''
    In Greek mythology, there are multiple stories associated with the constellation Cancer, but one prominent tale involves the second labor of Heracles (Hercules in Roman mythology). Hera, the wife of Zeus and the goddess of marriage, held a grudge against Heracles because he was the illegitimate son of Zeus and another woman. To harm Heracles, Hera sent a giant crab named Karkinos to distract him during his battle with the Hydra, a serpent with multiple heads.

    As Heracles was fighting the Hydra, Karkinos latched onto his foot with its strong pincers. However, Heracles quickly crushed the crab with his foot, killing it. In recognition of Karkinos' loyalty and sacrifice, Hera placed the crab in the night sky as the constellation Cancer. This was her way of honoring the creature that tried to thwart Heracles in his quest. The Cancer constellation is often depicted as a crab in various interpretations of this myth.

    '''

    print(len(text_to_infer))

    processor = AutoProcessor.from_pretrained(
        CFG.MODEL_NAME,
        voice_preset=CFG.SPEAKER,
        return_tensors='pt'
    )

    model = AutoModel.from_pretrained(
        CFG.MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "mps" or device.type == "cuda" else torch.float32,
    ).to(CFG.DEVICE)

    # # Modify model loading based on CUDA availability
    # if USE_CUDA:
    #     model = AutoModel.from_pretrained(
    #         CFG.MODEL_NAME,
    #         torch_dtype=torch.float16,  # half-precision
    #     ).to(CFG.DEVICE)
    #     # Only enable CPU offload if CUDA is available
    #     model.enable_cpu_offload()
    # else:
    #     print("model configured with CPU")
    #     # For CPU, avoid half-precision as it might not be well-supported
    #     model = AutoModel.from_pretrained(
    #         CFG.MODEL_NAME,
    #         torch_dtype=torch.float32,  # Use full precision for CPU
    #     ).to(CFG.DEVICE)

    ### inference optimization with accelerate
    model.enable_cpu_offload()

    n_params = count_parameters(model)
    print(n_params)

    total_tokens = count_tokens(text_to_infer, processor)
    print(total_tokens)

    sentences, number_of_sentences = split_sentences(text=text_to_infer)
    print(f'Sentences: \n {sentences}')
    print(f'\n\nNumber of sentences: {number_of_sentences}')

    ### prepare list of sentences
    sentences, number_of_sentences = split_sentences(text=text_to_infer)
    print(f'\nSentences in this text:\n {sentences}')
    print(f'\nNumber of sentences in this text: {number_of_sentences}\n')

    all_audio_arrays = []
    all_times = []

    ### inference per sentence
    for sentence_number in range(number_of_sentences):

        current_sentence = sentences[sentence_number]

        print(f'Processing sentence {sentence_number + 1}/{number_of_sentences}...')

        start_time = time.time()

        ### prepare input for the model, call the processor for the current sentence only
        inputs = processor(
            text=current_sentence,
            return_tensors="pt",
            return_attention_mask=True,
            max_length=1024,
            voice_preset=CFG.SPEAKER,
            add_special_tokens=False,
        ).to(CFG.DEVICE)

        ### count tokens
        n_tokens = count_tokens(current_sentence, processor)

        ### model inference
        with torch.inference_mode():
            result = model.generate(
                **inputs,
                do_sample=True,
                semantic_max_new_tokens=1024,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        ### save results
        all_audio_arrays.append(result.cpu().numpy())
        elapsed_time = round((time.time() - start_time), 2)
        all_times.append(elapsed_time)

        sentences_left_to_processs = number_of_sentences - (sentence_number + 1)
        average_time = np.array(all_times).mean()
        time_to_complete = round((sentences_left_to_processs * average_time) / 60, 2)  # in minutes

        print(f'''
              Sentence {sentence_number + 1}/{number_of_sentences} processed:
              \tNumber of tokens in sentence: {n_tokens}
              \tLength of sentence: {len(current_sentence)}
              \tNumber of sentences in text: {number_of_sentences}
              \tShape of tensor for this sentence: {result.size()}
              \tElapsed time for this sentence: {elapsed_time} s
              \tEstimated time to complete: {time_to_complete} min
              ''')

        sample_rate = model.generation_config.sample_rate  # 24_000

        result_array = result.cpu().numpy()
        result_len = result_array.shape[-1]

        fig, ax1 = plt.subplots(figsize=(15, 4))
        ax2 = ax1.twiny()
        print(current_sentence)

        ### plot audio array for this sentence
        values = result_array.ravel()
        # display(Audio(data=values, rate=sample_rate))
        write_wav(f"./data/test/test-sentence-{sentence_number}.wav", sample_rate, values)
        ax1.plot(values)

        ### calculate rolling amplitude Series
        rolling_window_size = sample_rate  # 24_000
        rolling_amplitude = pd.Series(np.abs(values)).rolling(window=rolling_window_size, min_periods=1).mean()

        ### plot the rolling amplitude Series with colors based on the rolling amplitude threshold
        rolling_threshold = CFG.AMPLITUDE_THRESHOLD  # 0.05
        ax1.plot(np.where(rolling_amplitude < rolling_threshold, rolling_amplitude, np.nan), color='orange',
                 label=f'Below Rolling Threshold (X={rolling_threshold})')
        ax1.plot(np.where(rolling_amplitude >= rolling_threshold, rolling_amplitude, np.nan), color='purple',
                 label=f'Above Rolling Threshold (X={rolling_threshold})')
        ax1.axhline(y=rolling_threshold, color='red', linestyle='--',
                    label=f'Rolling Threshold (X={rolling_threshold})')

        ### plot auxiliary X axis with Time in seconds: half-second positions
        half_second_positions = np.arange(0, result_len, sample_rate // 2)
        ax2.set_xticks(half_second_positions)
        ax2.set_xticklabels([f'{t:.1f}s' for t in half_second_positions / sample_rate])
        ax2.set_xlabel('Time (seconds)')
        for position in half_second_positions:
            plt.axvline(x=position, color='black', linestyle='-', linewidth=0.5)

        ### plot design
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlim(0, result_array.shape[-1])
        ax2.set_xlim(0, result_array.shape[-1])
        ax1.set_ylim(-0.20, 0.20)
        ax1.set_title(f'Audio {sentence_number + 1}/{number_of_sentences}')
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
        plt.show()

        del result
        gc.collect()

    avg_time = np.array(all_times).mean().round(2)  # average time per sentence (in seconds)
    print(avg_time)

    plt.figure(figsize=(8, 3))
    plt.hist(all_times, bins=len(sentences), color='blue', edgecolor='black')
    plt.title(f'Distribution of Processing Time for Sentences (total = {len(sentences)})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')

    ### Concatenate results for each batch
    concatenated_array = np.array([])

    for audio_number, sentence_audio in enumerate(all_audio_arrays):
        print(f'Audio {audio_number + 1}/{len(all_audio_arrays)}')

        ### concat audio arrays
        current_array = all_audio_arrays[audio_number].squeeze()

        ### post-processed array (remove padding in inference was done in batches)
        current_array = slice_array_wave(
            input_array=current_array,
            amplitude_threshold=CFG.AMPLITUDE_THRESHOLD,
            time_threshold=CFG.TIME_THRESHOLD,
            ignore_initial_steps=CFG.IGNORE_INITIAL_STEPS
        )

        concatenated_array = np.concatenate([concatenated_array, current_array])

    ### display the concatenated audio
    # display(Audio(data=concatenated_array, rate=sample_rate)) # normal speed
    write_wav(f"./data/test/test-all.wav", sample_rate, concatenated_array)

    ### save as np.array
    np.save(f'./final_audio_sr_{sample_rate}.npy', concatenated_array)

    ### save as .wav file
    write(
        f'./final_audio_sr_{sample_rate}.wav',
        rate=sample_rate,
        data=concatenated_array
    )


def check():
    sentence = '''
    It takes courage to grow up and become who you really are.
    '''.strip()
    prompts_folder = "bark/assets/prompts/"
    speaker_list = [f"{s}".replace('.npz', '') for s in os.listdir(prompts_folder) if s.endswith('.npz') and s.startswith('en_')]
    for i, speaker in enumerate(speaker_list):
        voice = generate_audio(sentence, history_prompt=speaker)
        write_wav(f"data/voice/{speaker.replace('/', '_')}.wav", SAMPLE_RATE, np.array(voice))

if __name__ == "__main__":
    preload_models()
    check()
