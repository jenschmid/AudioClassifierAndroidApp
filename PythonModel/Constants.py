"""
This file contains all constants and helper functions that are used in the different scripts.
"""
import numpy as np

# The following parameters must have the same value as in the front end
# For more information, please check the README
SAMPLE_RATE = 16000  # The sample rate of all audios used in the project
AUDIO_PIECE_LENGTH = SAMPLE_RATE * 2  # The length of each audio piece as specified by the task
N_MELS = 128  # The number of mels used in the mel spectrograms
IS_MONO = False  # Since all audios used in this project are converted to mono (not stereo) we set this boolean to False


def cut_audio(audio):
    """
    This method cuts an audio sample into pieces with length AUDIO_PIECE_LENGTH.
    The remainder of an audio sample that does not have length AUDIO_PIECE_LENGTH is omitted.

    :param audio: An audio signal in the form or a one-dimensional numeric array (length does not matter)

    :return audio_pieces: A list of audio sample pieces of length AUDIO_PIECE_LENGTH
    """
    start = 0
    end = len(audio)

    audio_pieces = []

    while start + AUDIO_PIECE_LENGTH < end:
        audio_pieces.append(audio[start:start + AUDIO_PIECE_LENGTH])
        start += AUDIO_PIECE_LENGTH

    return audio_pieces


def cut_audio_pad_last(audio):
    """
    This method cuts an audio sample into pieces with length AUDIO_PIECE_LENGTH.
    The remainder of an audio sample that does not have length AUDIO_PIECE_LENGTH is padded with zeros and included.

    :param audio: AAn audio signal in the form or a one-dimensional numeric array (length does not matter)

    :return audio_pieces: A list of audio sample pieces of length AUDIO_PIECE_LENGTH
    """
    start = 0
    end = len(audio)

    audio_pieces = []

    while start + AUDIO_PIECE_LENGTH < end:
        audio_pieces.append(audio[start:start + AUDIO_PIECE_LENGTH])
        start += AUDIO_PIECE_LENGTH

    # Pad the remainder of an audio sample with zeroes to match the length of all pieces and add it to the list
    last_piece = audio[start:end]
    last_piece = np.pad(last_piece, (0, AUDIO_PIECE_LENGTH - (end - start)), 'constant')
    audio_pieces.append(last_piece)

    return audio_pieces
