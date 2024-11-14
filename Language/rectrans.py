# pip install numpy sounddevice webrtcvad playsound keyboard google-cloud-speech google-cloud-texttospeech google-cloud-translate


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
from playsound import playsound  # For audio playback
from pydub import AudioSegment
from pydub.playback import play
import keyboard  # For detecting key presses

# Set your environment variable for Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\andre\\Desktop\\language-440220-e0110f2acfe7.json'

# Audio recording parameters
SAMPLE_RATE = 16000  # Recommended sample rate for Google Speech-to-Text
FRAME_DURATION = 30  # Frame duration in milliseconds (10, 20, or 30 ms)
NUM_CHANNELS = 1
VAD_MODE = 3  # Aggressiveness mode (0-3)

# Initialize VAD
vad = webrtcvad.Vad(VAD_MODE)

# Global variables for language settings
base_language = 'en-US'  # User's base language
lang_combos = [
    [base_language, 'es-ES'],  # English-Spanish
    [base_language, 'ko-KR'],   # English-Korean
    [base_language, 'en-US']
]
mode_index = -1  # Index to keep track of the current language pair
mode = None  # set initial mode to None

# Lock for thread-safe operations
language_lock = threading.Lock()

def read_audio_chunk(stream, frame_duration, sample_rate):
    """Read a chunk of audio from the stream."""
    n_frames = int(sample_rate * (frame_duration / 1000.0))
    try:
        audio, _ = stream.read(n_frames)  # Unpack the tuple to get only the audio data
        return audio
    except Exception as e:
        print(f"Error reading audio: {e}")
        return None

# function to detect voice activity
def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, stream):
    """Yield segments of audio where speech is detected."""
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    print("VAD collector started. Listening for voice activity...")

    while True:
        audio = read_audio_chunk(stream, frame_duration_ms, sample_rate)
        if audio is None:
            continue

        # Convert audio to bytes before passing to VAD
        is_speech = vad.is_speech(audio.tobytes(), sample_rate)

        if is_speech:
            if not triggered:
                print("Voice activity detected.")
                triggered = True
                voiced_frames.append(audio)
            else:
                voiced_frames.append(audio)
            ring_buffer.clear()
        else:
            if triggered:
                ring_buffer.append(audio)
                if len(ring_buffer) >= ring_buffer.maxlen:
                    print("End of voice activity detected.")
                    # Concatenate voiced frames and yield
                    yield b''.join([f.tobytes() for f in voiced_frames])
                    # Reset for next utterance
                    triggered = False
                    voiced_frames = []
                    ring_buffer.clear()
            else:
                continue  # Stay in silence until voice is detected

def translate_text(text, target_language):
    """Translate the text to the target language using Google Cloud Translation API."""
    print(f"Initializing Translation client...")
    translate_client = translate.Client()

    # Translate text
    print(f"Translating text to {target_language}...")
    result = translate_client.translate(text, target_language=target_language)
    translated_text = result["translatedText"]
    print(f"Translated Text: {translated_text}")
    return translated_text

def transcribe_and_translate(audio_bytes):
    """Transcribe the audio, detect language, set mode, translate, and synthesize the translated text."""
    global mode, mode_index

    print("Initializing Speech-to-Text client...")
    client = speech.SpeechClient()

    # Configure transcription settings with alternative languages
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code=base_language,  # Base language
        sample_rate_hertz=SAMPLE_RATE,
        alternative_language_codes=["es-ES", "ko-KR"]  # Additional languages
    )

    # Perform transcription
    print("Transcribing audio...")
    response = client.recognize(config=config, audio=audio)

    # Check if any results were returned
    if not response.results:
        print("No transcription results found.")
        return

    # Process transcription results
    full_transcript = ""
    for result in response.results:
        # Each result is for a consecutive portion of the audio.
        alternative = result.alternatives[0]
        transcript = alternative.transcript
        full_transcript += transcript + " "

    print(f"Transcription result: {full_transcript.strip()}")

    # Use Translation API to detect the language of the transcribed text
    print("Detecting language of the transcribed text...")
    translate_client = translate.Client()
    detection = translate_client.detect_language(full_transcript)
    detected_language = detection['language']
    print(f"Detected language: {detected_language}")

    with language_lock:            
        # Check if mode is set or if detected language is neither base nor current target
        if mode is None or (detected_language != base_language[:2] and detected_language != mode[1][:2]):
            # Look for a language pair with the base language as English and the target as the detected language
            found_pair = None
            for pair in lang_combos:
                if pair[0] == base_language and pair[1][:2] == detected_language:
                    found_pair = pair
                    break
            
            if found_pair:
                mode = found_pair
                print(f"Mode set to detected language pair: {mode}")
            else:
                print(f"No matching language pair found for detected language ({detected_language}) with English base.")
                return  # Do nothing if no suitable language combo is found

        # Set the target language based on the current mode
        target_language = mode[1] if detected_language == mode[0][:2] else mode[0]
        print(f"Mode set to: {mode}")

        # No swapping if detected language is the same as target language
        if detected_language == target_language[:2]:
            print("Detected language is the same as target language, proceeding with translation without swapping.")

    # Translate the text
    translated_text = translate_text(full_transcript, target_language[:2])
    # Synthesize speech
    synthesize_speech(translated_text, target_language)
    # Play the synthesized speech
    print("Playing the synthesized speech...")
    print("Playback finished.")

def synthesize_speech(text, target_language_code):
    """Convert text to speech and save it as a WAV file, then play the file."""
    tts_client = texttospeech.TextToSpeechClient()

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
    print("Synthesizing speech from text...")
    response = tts_client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

    # Save the audio file as WAV
    output_audio_file = "translated_audio.wav"
    with open(output_audio_file, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio content written to file '{output_audio_file}'")

    # Play the synthesized speech using pydub
    print("Playing the synthesized speech...")
    audio = AudioSegment.from_wav(output_audio_file)
    play(audio)
    print("Playback finished.")


def switch_mode():
    """Switch to the next language mode."""
    global mode_index, mode
    with language_lock:
        mode_index = (mode_index + 1) % len(lang_combos)
        mode = lang_combos[mode_index]
        print(f"Manually switched mode to: {mode}")

def reset_mode():
    """Reset to the original language mode."""
    global mode_index, mode, base_language
    with language_lock:
        mode_index = -1
        mode = None
        base_language = 'en-US'
        print("Reset to original mode: English as base language with autodetection.")

def listen_for_key_presses():
    """Listen for 's' and 'r' key presses to switch or reset languages."""
    print("Key listener thread started. Waiting for key presses ('s' to switch, 'r' to reset)...")
    while True:
        try:
            if keyboard.is_pressed('s'):
                print("'s' key pressed. Switching language mode...")
                switch_mode()
                time.sleep(0.5)  # Prevent multiple detections
            elif keyboard.is_pressed('r'):
                print("'r' key pressed. Resetting language mode...")
                reset_mode()
                time.sleep(0.5)  # Prevent multiple detections
            time.sleep(0.1)
        except:
            break

def main():
    print("Starting automatic translator. Press 's' to switch languages, 'r' to reset, or Ctrl+C to stop.")

    # Start a thread to listen for key presses
    key_listener_thread = threading.Thread(target=listen_for_key_presses, daemon=True)
    key_listener_thread.start()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=NUM_CHANNELS, dtype='int16') as stream:
            print("Audio input stream opened.")
            while True:
                current_base_language = base_language  # Capture the current base language
                print(f"\nListening for speech in: {current_base_language} (Mode: {mode})")

                # Collect voiced audio frames
                frames = vad_collector(SAMPLE_RATE, FRAME_DURATION, padding_duration_ms=300, vad=vad, stream=stream)

                for audio_data in frames:
                    print("Processing captured voice data...")
                    # Save the recorded audio to a WAV file (optional)
                    recorded_file_path = "Recording.wav"
                    with wave.open(recorded_file_path, 'wb') as wf:
                        wf.setnchannels(NUM_CHANNELS)
                        wf.setsampwidth(2)  # Since dtype='int16', the sample width is 2 bytes
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes(audio_data)
                    print(f"Recorded audio saved to '{recorded_file_path}'")

                    # Transcribe, translate, and synthesize speech
                    transcribe_and_translate(audio_data)

                    # Break if base_language changes during processing (due to key press)
                    if base_language != current_base_language:
                        print("Base language changed during processing. Restarting listening loop.")
                        break

    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit()

if __name__ == "__main__":
    main()
