from flask import Flask, request, render_template_string, send_file
from google.cloud import storage, speech, texttospeech, translate_v2 as translate
import os
import requests
import io

app = Flask(__name__)

# Set the environment variable for Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\Owner\\Downloads\\portable-language-translator-8acaf47b0c5f.json'

# Initialize Google Cloud clients
storage_client = storage.Client()
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()
translate_client = translate.Client()

# Your bucket name
BUCKET_NAME = 'bucket-quickstart_portable-language-translator'

# Home route
@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Home Page!"

# Transcribe audio function
def transcribe_audio(file_name):
    audio = speech.RecognitionAudio(uri=f'gs://{BUCKET_NAME}/{file_name}')
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code='en-US',
    )
    response = speech_client.recognize(config=config, audio=audio)
    transcriptions = [result.alternatives[0].transcript for result in response.results]
    return ' '.join(transcriptions)

# Synthesize speech function (Text-to-Speech)
def synthesize_speech(text):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="es",  # Change to Spanish for TTS output
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    audio_output = io.BytesIO(response.audio_content)
    audio_output.seek(0)  # Reset the buffer pointer
    return audio_output

# Upload audio file route (for transcription and synthesis)
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            bucket = storage_client.get_bucket(BUCKET_NAME)
            blob = bucket.blob(file.filename)
            blob.upload_from_file(file)
            
            transcription = transcribe_audio(blob.name)
            audio_output = synthesize_speech(transcription)
            return send_file(audio_output, mimetype='audio/mpeg', as_attachment=True, download_name='output.mp3')

    return '''
    <h1>Upload File</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <input type="submit" value="Upload">
    </form>
    '''

# Upload text file route (for Text-to-Speech)
@app.route('/upload_text', methods=['GET', 'POST'])
def upload_text_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.txt'):
            # Upload the text file to Google Cloud Storage
            bucket = storage_client.get_bucket(BUCKET_NAME)
            blob = bucket.blob(file.filename)
            blob.upload_from_file(file)

            # Read the text content from the file
            text_content = blob.download_as_text()

            # Synthesize the text content into speech
            audio_output = synthesize_speech(text_content)

            # Return the synthesized audio file for download
            return send_file(audio_output, mimetype='audio/mpeg', as_attachment=True, download_name='text_to_speech_output.mp3')

    return '''
    <h1>Upload Text File for Speech Synthesis</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".txt" required>
        <input type="submit" value="Upload and Convert to Speech">
    </form>
    '''

# Upload combined route (audio upload, transcription, translation, and synthesis)
@app.route('/upload_combined', methods=['GET', 'POST'])
def upload_combined():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            bucket = storage_client.get_bucket(BUCKET_NAME)
            blob = bucket.blob(file.filename)
            blob.upload_from_file(file)

            # Transcribe the audio file
            transcription = transcribe_audio(blob.name)

            # Translate the transcribed text to Spanish
            translation = translate_client.translate(transcription, target_language='es')
            translated_text = translation['translatedText']

            # Synthesize the translated Spanish text into speech
            audio_output = synthesize_speech(translated_text)

            # Return the synthesized audio file for download
            return send_file(audio_output, mimetype='audio/mpeg', as_attachment=True, download_name='translated_speech.mp3')

    return '''
    <h1>Upload Audio File for Combined Functionality</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <input type="submit" value="Upload and Process">
    </form>
    '''

# API Testing route (testing audio file upload)
@app.route('/test_upload', methods=['POST'])
def test_upload():
    test_file_path = 'C:\\Users\\Owner\\Documents\\Google Cloud Testing\\Uploads\\You want to play Let.wav'
    with open(test_file_path, 'rb') as test_file:
        files = {'file': test_file}
        response = requests.post(f'http://127.0.0.1:5000/upload_combined', files=files)
    return f'Test Upload Response: {response.text}'

if __name__ == '__main__':
    app.run(debug=True)
