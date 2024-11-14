# translator_device.py

import os
import io
import wave
import numpy as np
import sounddevice as sd
from google.cloud import speech
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate
import webrtcvad  # VAD library
import collections
import sys
import threading
import time
from pydub import AudioSegment
from pydub.playback import play
from pynput import keyboard  # For detecting key presses

# Set your environment variable for Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\andre\\Desktop\\language-440220-e0110f2acfe7.json'

class TranslatorDevice:
    def __init__(self):
        # Audio recording parameters
        self.SAMPLE_RATE = 16000  # Recommended sample rate for Google Speech-to-Text
        self.FRAME_DURATION = 30  # Frame duration in milliseconds (10, 20, or 30 ms)
        self.NUM_CHANNELS = 1
        self.VAD_MODE = 3  # Aggressiveness mode (0-3)

        # Initialize VAD
        self.vad = webrtcvad.Vad(self.VAD_MODE)

        # Language settings
        self.base_language = 'en-US'  # User's base language
        self.lang_combos = [
            [self.base_language, 'es-ES'],  # English-Spanish
            [self.base_language, 'ko-KR'],   # English-Korean
            [self.base_language, 'en-US']
        ]
        self.mode_index = -1  # Index to keep track of the current language pair
        self.mode = None  # Set initial mode to None

        # Lock for thread-safe operations
        self.language_lock = threading.Lock()

        # Initialize Google Cloud clients
        self.speech_client = speech.SpeechClient()
        self.translate_client = translate.Client()
        self.tts_client = texttospeech.TextToSpeechClient()

        # Key press handling
        self.listener = None
        self.switch_mode_key_pressed = False
        self.reset_mode_key_pressed = False

    def read_audio_chunk(self, stream, frame_duration, sample_rate):
        """Read a chunk of audio from the stream."""
        n_frames = int(sample_rate * (frame_duration / 1000.0))
        try:
            audio, _ = stream.read(n_frames)  # Unpack the tuple to get only the audio data
            return audio
        except Exception as e:
            # print(f"Error reading audio: {e}")
            return None

    def vad_collector(self, sample_rate, frame_duration_ms, padding_duration_ms, stream):
        """Yield segments of audio where speech is detected."""
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        voiced_frames = []
        # print("VAD collector started. Listening for voice activity...")

        while True:
            audio = self.read_audio_chunk(stream, frame_duration_ms, sample_rate)
            if audio is None:
                continue

            # Convert audio to bytes before passing to VAD
            is_speech = self.vad.is_speech(audio.tobytes(), sample_rate)

            if is_speech:
                if not triggered:
                    # print("Voice activity detected.")
                    triggered = True
                    voiced_frames.append(audio)
                else:
                    voiced_frames.append(audio)
                ring_buffer.clear()
            else:
                if triggered:
                    ring_buffer.append(audio)
                    if len(ring_buffer) >= ring_buffer.maxlen:
                        # print("End of voice activity detected.")
                        # Concatenate voiced frames and yield
                        yield b''.join([f.tobytes() for f in voiced_frames])
                        # Reset for next utterance
                        triggered = False
                        voiced_frames = []
                        ring_buffer.clear()
                else:
                    continue  # Stay in silence until voice is detected

    def translate_text(self, text, target_language):
        """Translate the text to the target language using Google Cloud Translation API."""
        # print(f"Translating text to {target_language}...")
        result = self.translate_client.translate(text, target_language=target_language)
        translated_text = result["translatedText"]
        # print(f"Translated Text: {translated_text}")
        return translated_text

    def transcribe_and_translate(self, audio_bytes):
        """Transcribe the audio, detect language, set mode, translate, and synthesize the translated text."""
        # print("Transcribing audio...")
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code=self.base_language,  # Base language
            sample_rate_hertz=self.SAMPLE_RATE,
            alternative_language_codes=["es-ES", "ko-KR"]  # Additional languages
        )

        # Perform transcription
        response = self.speech_client.recognize(config=config, audio=audio)

        # Check if any results were returned
        if not response.results:
            # print("No transcription results found.")
            return

        # Process transcription results
        full_transcript = ""
        for result in response.results:
            # Each result is for a consecutive portion of the audio.
            alternative = result.alternatives[0]
            transcript = alternative.transcript
            full_transcript += transcript + " "

        full_transcript = full_transcript.strip()
        # print(f"Transcription result: {full_transcript}")

        # Detect language of the transcribed text
        # print("Detecting language of the transcribed text...")
        detection = self.translate_client.detect_language(full_transcript)
        detected_language = detection['language']
        # print(f"Detected language: {detected_language}")

        with self.language_lock:
            # Check if mode is set or if detected language is neither base nor current target
            if self.mode is None or (detected_language != self.base_language[:2] and detected_language != self.mode[1][:2]):
                # Look for a language pair with the base language and the detected language
                found_pair = None
                for pair in self.lang_combos:
                    if pair[0][:2] == self.base_language[:2] and pair[1][:2] == detected_language:
                        found_pair = pair
                        break

                if found_pair:
                    self.mode = found_pair
                    # print(f"Mode set to detected language pair: {self.mode}")
                else:
                    # print(f"No matching language pair found for detected language ({detected_language}) with base language.")
                    return  # Do nothing if no suitable language combo is found

            # Set the target language based on the current mode
            if detected_language == self.mode[0][:2]:
                target_language = self.mode[1]
            else:
                target_language = self.mode[0]

            # print(f"Mode set to: {self.mode}")

        # Translate the text
        translated_text = self.translate_text(full_transcript, target_language[:2])

        # Synthesize speech
        self.synthesize_speech(translated_text, target_language)

    def synthesize_speech(self, text, target_language_code):
        """Convert text to speech and play the audio without saving to a file."""
        # Select appropriate voice parameters
        voice_params = {
            "es": ("es-ES", texttospeech.SsmlVoiceGender.NEUTRAL),
            "en": ("en-US", texttospeech.SsmlVoiceGender.NEUTRAL),
            "ko": ("ko-KR", texttospeech.SsmlVoiceGender.NEUTRAL)
        }
        language_code, ssml_gender = voice_params.get(target_language_code[:2], ("en-US", texttospeech.SsmlVoiceGender.NEUTRAL))

        # Configure the text input and voice parameters
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            ssml_gender=ssml_gender
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16  # LINEAR16 is WAV format
        )

        # Generate the speech
        # print("Synthesizing speech from text...")
        response = self.tts_client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

        # Play the synthesized speech without saving to a file
        # print("Playing the synthesized speech...")
        audio_content = response.audio_content
        audio_stream = io.BytesIO(audio_content)
        audio_segment = AudioSegment.from_file(audio_stream, format="wav")
        play(audio_segment)
        # print("Playback finished.")

    # def switch_mode(self):
    #     """Switch to the next language mode."""
    #     with self.language_lock:
    #         self.mode_index = (self.mode_index + 1) % len(self.lang_combos)
    #         self.mode = self.lang_combos[self.mode_index]
    #         # print(f"Manually switched mode to: {self.mode}")

    # def reset_mode(self):
    #     """Reset to the original language mode."""
    #     with self.language_lock:
    #         self.mode_index = -1
    #         self.mode = None
    #         self.base_language = 'en-US'
            # print("Reset to original mode: English as base language with autodetection.")

    # def on_press(self, key):
    #     """Handle key press events."""
    #     try:
    #         if key.char == 's':
    #             # print("'s' key pressed. Switching language mode...")
    #             self.switch_mode()
    #         elif key.char == 'r':
    #             # print("'r' key pressed. Resetting language mode...")
    #             self.reset_mode()
    #     except AttributeError:
    #         pass  # Special keys (e.g., Ctrl, Alt) are ignored

    # def start_key_listener(self):
    #     """Start the key listener thread."""
    #     self.listener = keyboard.Listener(on_press=self.on_press)
    #     self.listener.start()

    def start(self):
        # print("Starting automatic translator. Press 's' to switch languages, 'r' to reset, or Ctrl+C to stop.")

        # Start key listener
        # self.start_key_listener()

        try:
            with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=self.NUM_CHANNELS, dtype='int16') as stream:
                # print("Audio input stream opened.")
                while True:
                    current_base_language = self.base_language  # Capture the current base language
                    print(f"\nListening for speech in: {current_base_language} (Mode: {self.mode})")

                    # Collect voiced audio frames
                    frames_generator = self.vad_collector(
                        self.SAMPLE_RATE,
                        self.FRAME_DURATION,
                        padding_duration_ms=300,
                        stream=stream
                    )

                    for audio_data in frames_generator:
                        # print("Processing captured voice data...")

                        # Transcribe, translate, and synthesize speech
                        self.transcribe_and_translate(audio_data)

                        # Break if base_language changes during processing (due to key press)
                        if self.base_language != current_base_language:
                            # print("Base language changed during processing. Restarting listening loop.")
                            break

        except KeyboardInterrupt:
            # print("\nExiting...")
            if self.listener:
                self.listener.stop()
            sys.exit()
