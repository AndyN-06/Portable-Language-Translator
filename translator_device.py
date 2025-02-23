# translator_device.py

import os
import io
import sys
import threading
import collections
import numpy as np
import sounddevice as sd
from google.cloud import speech
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate
import webrtcvad  # Voice Activity Detection library
from pydub import AudioSegment
# from pydub.playback import play
import time
from pydub.playback import _play_with_simpleaudio as play

# Set your environment variable for Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/plt/Desktop/optimum-reactor-449320-e8-dcb220f309a5.json'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "C:\\Users\\yohan\\Desktop\\optimum-reactor-449320-e8-dcb220f309a5.json"

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
        self.base_language = 'en-US'  # Default base language
        self.supported_languages = ['en-US', 'es-US', 'ko-KR']  # Supported languages
        self.lang_combos = [
            [self.base_language, 'es-US'],  # Base-Spanish
            [self.base_language, 'ko-KR'],  # Base-Korean
            [self.base_language, 'en-US']   # Base-English (self-translation)
        ]
        self.mode = None  # Initial mode

        # Voice settings
        self.gender = 'NEUTRAL'  # Default gender
        self.voice_type = 'Standard'  # Default type

        # Lock for thread-safe operations
        self.language_lock = threading.Lock()

        # Initialize Google Cloud clients
        self.speech_client = speech.SpeechClient()
        self.translate_client = translate.Client()
        self.tts_client = texttospeech.TextToSpeechClient()

        # Active flag to control processing. When False, the device is "paused".
        self.active = True
        self.vad_active = True
        self.reset_time = None
        

    def read_audio_chunk(self, stream, frame_duration, sample_rate):
        """Read a chunk of audio from the stream."""
        n_frames = int(sample_rate * (frame_duration / 1000.0))
        try:
            audio, _ = stream.read(n_frames)  # Unpack the tuple to get only the audio data
            return audio
        except Exception as e:
            print(f"Error reading audio: {e}")
            return None

    def vad_collector(self, sample_rate, frame_duration_ms, padding_duration_ms, stream):
        """Yield segments of audio where speech is detected."""
        voiced_frames = []
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        voiced_frames = []

        while True:
            # If the device is paused, break out of this generator.
            if not self.vad_active:
                break

            audio = self.read_audio_chunk(stream, frame_duration_ms, sample_rate)
            if audio is None:
                continue

            # Convert audio to bytes before passing to VAD
            is_speech = self.vad.is_speech(audio.tobytes(), sample_rate)

            if is_speech:
                if not triggered:
                    triggered = True
                    voiced_frames.append(audio)
                else:
                    voiced_frames.append(audio)
                ring_buffer.clear()
            else:
                if triggered:
                    ring_buffer.append(audio)
                    if len(ring_buffer) >= ring_buffer.maxlen:
                        # Concatenate voiced frames and yield
                        yield b''.join([f.tobytes() for f in voiced_frames])
                        # Reset for next utterance
                        triggered = False
                        voiced_frames = []
                        ring_buffer.clear()
                else:
                    continue  # Remain in silence until voice is detected

    def translate_text(self, text, target_language):
        """Translate the text to the target language using Google Cloud Translation API."""
        result = self.translate_client.translate(text, target_language=target_language)
        translated_text = result["translatedText"]
        return translated_text

    def transcribe_and_translate(self, audio_bytes):
        """Transcribe the audio, detect language, set mode, translate, and synthesize the translated text."""
        start_time = time.time()

        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code=self.base_language,  # Base language
            sample_rate_hertz=self.SAMPLE_RATE,
            alternative_language_codes=[lang for lang in self.supported_languages if lang != self.base_language]
        )

        response = self.speech_client.recognize(config=config, audio=audio)

        if not response.results:
            return

        full_transcript = ""
        for result in response.results:
            alternative = result.alternatives[0]
            transcript = alternative.transcript
            full_transcript += transcript + " "

        full_transcript = full_transcript.strip()
        print(f"Transcription result: {full_transcript}")

        detection = self.translate_client.detect_language(full_transcript)
        detected_language = detection['language']

        with self.language_lock:
            if self.mode is None or (detected_language != self.base_language[:2] and detected_language != self.mode[1][:2]):
                found_pair = None
                for pair in self.lang_combos:
                    if pair[0][:2] == self.base_language[:2] and pair[1][:2] == detected_language:
                        found_pair = pair
                        break

                if found_pair:
                    self.mode = found_pair
                else:
                    return

            if detected_language == self.mode[0][:2]:
                target_language = self.mode[1]
            else:
                target_language = self.mode[0]

        translated_text = self.translate_text(full_transcript, target_language[:2])
        print(f"Translated text: {translated_text}")

        playback_start_time = time.time()
        print(f"Total time from sending audio to playback: {playback_start_time - start_time:.2f} seconds")

        self.synthesize_speech(translated_text, target_language)

    def get_voice_variant(self, language_code, ssml_gender):
        """Get the voice variant letter based on language code and gender."""
        voice_variants = {
            'en-US': {
                texttospeech.SsmlVoiceGender.FEMALE: 'F',
                texttospeech.SsmlVoiceGender.MALE: 'B',
            },
            'es-US': {
                texttospeech.SsmlVoiceGender.FEMALE: 'A',
                texttospeech.SsmlVoiceGender.MALE: 'C',
            },
            'ko-KR': {
                texttospeech.SsmlVoiceGender.FEMALE: 'B',
                texttospeech.SsmlVoiceGender.MALE: 'C',
            },
        }
        variant = voice_variants.get(language_code, {}).get(ssml_gender)
        if variant:
            return variant
        else:
            print(f"No variant found for {language_code} with gender {ssml_gender}. Using default variant 'A'.")
            return 'A'

    def synthesize_speech(self, text, target_language_code):
        """Convert text to speech and play the audio without saving to a file."""
        gender_map = {
            'MALE': texttospeech.SsmlVoiceGender.MALE,
            'FEMALE': texttospeech.SsmlVoiceGender.FEMALE
        }
        ssml_gender = gender_map.get(self.gender, texttospeech.SsmlVoiceGender.MALE)
        variant = self.get_voice_variant(target_language_code, ssml_gender)
        voice_name = f"{target_language_code}-Standard-{variant}"
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=target_language_code,
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )

        try:
            response = self.tts_client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
            audio_content = response.audio_content
            audio_stream = io.BytesIO(audio_content)
            audio_segment = AudioSegment.from_file(audio_stream, format="wav")
            play_obj = play(audio_segment)
            play_obj.wait_done()
        except Exception as e:
            print(f"Error during speech synthesis: {e}")

    def set_settings(self, base_language, gender):
        with self.language_lock:
            self.base_language = base_language
            self.gender = gender
            self.mode = None  # Reset mode
            self.lang_combos = [
                [self.base_language, lang] for lang in self.supported_languages if lang != self.base_language
            ]
            print(f"Settings updated: Base Language - {self.base_language}, Gender - {self.gender}")

    def start(self):
        print("Starting automatic translator device.")
        try:
            with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=self.NUM_CHANNELS, dtype='int16') as stream:
                print("Audio input stream opened.")
                while True:
                    # Pause processing if not active (speech mode is off)
                    if not self.active:
                        time.sleep(0.1)
                        continue

                    current_base_language = self.base_language
                    print(f"\nListening for speech in: {current_base_language} (Mode: {self.mode})")

                    frames_generator = self.vad_collector(
                        self.SAMPLE_RATE,
                        self.FRAME_DURATION,
                        padding_duration_ms=300,
                        stream=stream
                    )

                    for audio_data in frames_generator:
                        # If paused during processing, break out to recheck the active flag
                        if self.reset_time and time.time() < self.reset_time + 1:
                            print("Discarding residual audio segment due to recent mode switch...")
                            continue
                        
                        if not self.active:
                            break
                        print("Processing captured voice data...")
                        self.transcribe_and_translate(audio_data)
                        
                        if self.base_language != current_base_language:
                            print("Base language changed during processing. Restarting listening loop.")
                            break

        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit()

    def listen_and_save_transcription(self, file_path):
        """Listen until a complete utterance is detected using VAD,
        transcribe the audio for the base language, save the transcript to file, and return the transcript."""
        print("Listening for a voice utterance...")
        with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=self.NUM_CHANNELS, dtype='int16') as stream:
            # Use the VAD collector to yield audio segments.
            for audio_bytes in self.vad_collector(
                    self.SAMPLE_RATE,
                    self.FRAME_DURATION,
                    padding_duration_ms=300,
                    stream=stream):
                print("Processing captured voice data...")
                # Check if the audio segment is long enough to be meaningful.
                if len(audio_bytes) < 1000:
                    print("Audio segment too short, ignoring...")
                    continue

                audio = speech.RecognitionAudio(content=audio_bytes)
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    language_code=self.base_language,  # Only listen for the base language
                    sample_rate_hertz=self.SAMPLE_RATE
                )
                try:
                    response = self.speech_client.recognize(config=config, audio=audio)
                    transcript = " ".join(
                        [result.alternatives[0].transcript for result in response.results]
                    ).strip()
                    if not transcript:
                        print("No speech detected in this segment, waiting for valid input...")
                        continue
                    print(f"Transcription recorded: {transcript}")
                except Exception as e:
                    transcript = ""
                    print(f"Error transcribing audio: {e}")
                    continue

                # Save transcript to a file.
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(transcript)
                    print(f"Transcription saved to {file_path}")
                except Exception as e:
                    print(f"Error writing transcription to file: {e}")

                return transcript


    def reset(self):
        self.reset_time = time.time()
        print("Translator device reset: Ignoring audio for the next 2 seconds.")

