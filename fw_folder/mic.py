# import sounddevice as sd
# import numpy as np

# duration = 5  # seconds
# sample_rate = 44100  # Hz

# print("Recording...")
# audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
# sd.wait()  # Wait for the recording to complete

# print("Recording complete. Saving to file...")
# # Save as a .wav file
# import scipy.io.wavfile as wavfile
# wavfile.write('output.wav', sample_rate, audio_data)
# print("File saved as output.wav")

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile

class AudioRecorder:
    def __init__(self, duration=5, sample_rate=44100, file_name="output.wav"):
        self.duration = duration
        self.sample_rate = sample_rate
        self.file_name = file_name
        self.audio_data = None

    def record(self):
        print("Recording...")
        self.audio_data = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16'
        )
        sd.wait()  # Wait for the recording to complete
        print("Recording complete.")

    def save_to_file(self):
        if self.audio_data is not None:
            print("Saving to file...")
            wavfile.write(self.file_name, self.sample_rate, self.audio_data)
            print(f"File saved as {self.file_name}")
        else:
            print("No audio data to save. Please record first.")


if __name__ == "__main__":
    recorder = AudioRecorder(duration=5, sample_rate=44100, file_name="output.wav")
    recorder.record()
    recorder.save_to_file()
