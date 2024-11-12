import os
import sounddevice as sd                            # to get audio from device
from scipy.io.wavfile import write                  # write audio to wav file
from google.cloud import speech                     # speech to text api from google
import tempfile                                     # for temp file to hold audio file
import io
import keyboard                                     # Library to detect key presses
import numpy as np

# Set your environment variable for Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\andre\\Downloads\\language-440220-e0110f2acfe7.json'

# Google Cloud Speech client
client = speech.SpeechClient()

# Audio recording parameters
SAMPLE_RATE = 16000  # Google recommends 16kHz for speech recognition
is_recording = False
audio_data = []  # List to store audio data chunks

def start_recording():
    """Start recording audio when 'r' is pressed."""
    global is_recording, audio_data
    print("Recording started. Press 'r' again to stop.")
    is_recording = True
    audio_data = []
    while is_recording:
        chunk = sd.rec(int(SAMPLE_RATE * 0.1), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        audio_data.append(chunk)

def stop_recording_and_transcribe():
    """Stop recording and transcribe the recorded audio."""
    global is_recording, audio_data
    print("Recording stopped.")
    is_recording = False

    # Combine all chunks into a single numpy array
    recorded_audio = np.concatenate(audio_data, axis=0)

    # Save the audio data to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav:
        write(temp_wav.name, SAMPLE_RATE, recorded_audio)  # Write the audio data to the WAV file
        print(f"Audio saved temporarily to {temp_wav.name}")

        # Read the WAV file into memory
        with io.open(temp_wav.name, "rb") as audio_file:
            content = audio_file.read()

        # Set up Google Cloud Speech-to-Text configuration
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            
            # primary language
            language_code="en-US",

            # alternative languages
            alternative_language_codes=["es-ES", "ko-KR"]
        )

        # Perform transcription
        print("Transcribing audio...")
        response = client.recognize(config=config, audio=audio)
        
        # Display transcription results
        for result in response.results:
            print("Transcript: {}".format(result.alternatives[0].transcript))

def main():
    print("Press 'r' to start recording and 'r' again to stop.")
    while True:
        if keyboard.is_pressed("r"):
            global is_recording
            if not is_recording:
                start_recording()
            else:
                stop_recording_and_transcribe()
                break  # Exit after transcription

if __name__ == "__main__":
    main()