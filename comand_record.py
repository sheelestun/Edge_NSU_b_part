import numpy as np

import config as cfg


class CommandRecorder:

    def __init__(self):
        self.audio_chunks = []
        self.speech_started = False
        self.silence_chunks = 0
        self.finished = False

    def process(self, audio):
        self.audio_chunks.append(audio.copy())

        volume = np.abs(audio).mean()

        if volume > cfg.SILENCE_THRESHOLD:
            self.speech_started = True
            self.silence_chunks = 0

        elif self.speech_started:
            self.silence_chunks += 1

            if self.silence_chunks >= cfg.SILENCE_CHUNKS:
                self.finished = True

    def get_audio(self):
        return np.concatenate(self.audio_chunks)