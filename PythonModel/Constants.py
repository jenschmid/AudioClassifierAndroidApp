"""
This file contains all constants and helper functions that are used in the different scripts.
"""
import numpy as np

SAMPLE_RATE = 16000  # The sample rate of all audios used in the project
AUDIO_PIECE_LENGTH = SAMPLE_RATE * 2  # The length of each audio piece as specified by the task
N_MELS = 128  # The number of mels used in the mel spectrograms
IS_MONO = False  # Since all audios used in this project are converted to mono (not stereo) we set this boolean to False


def cut_audio(audio):
    """
    This method cuts an audio in pieces with length AUDIO_PIECE_LENGTH.
    The remainder of an audio that does not fit into such a piece is omitted.

    :param audio: An audio in the form or a one-dimensional numeric array (length does not matter)

    :return audio_pieces: A list of pieces of the audio of length AUDIO_PIECE_LENGTH
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
    This method cuts an audio in pieces with length AUDIO_PIECE_LENGTH.
    The remainder of an audio that does not fit into such a piece is padded by zeroes and added to the list.

    :param audio: An audio in the form or a one-dimensional numeric array (length does not matter)

    :return audio_pieces: A list of pieces of the audio of length AUDIO_PIECE_LENGTH
    """
    start = 0
    end = len(audio)

    audio_pieces = []

    while start + AUDIO_PIECE_LENGTH < end:
        audio_pieces.append(audio[start:start + AUDIO_PIECE_LENGTH])
        start += AUDIO_PIECE_LENGTH

    # Add the remainder of an audio and pad it to match the length of all pieces
    last_piece = audio[start:end]
    last_piece = np.pad(last_piece, (0, AUDIO_PIECE_LENGTH - (end - start)), 'constant')
    audio_pieces.append(last_piece)

    return audio_pieces
