import tempfile
import logging
from flask import Flask, request, render_template_string, jsonify
from google.cloud import storage, speech, texttospeech, translate_v2 as translate
import os
import io

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set the environment variable for Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\andre\\Desktop\\optimum-reactor-449320-e8-dcb220f309a5.json'

# Initialize Google Cloud clients
storage_client = storage.Client()
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()
translate_client = translate.Client()

# Your bucket name
BUCKET_NAME = 'portable-lang'

# Home route to render the HTML template
@app.route('/', methods=['GET'])
def home():
    HTML_TEMPLATE = '''
    <!doctype html>
    <html lang="en">
    <head>
        <title>Audio Translator</title>
        <script>
            let mediaRecorder;
            let audioChunks = [];
            
            function startRecording() {
                navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
                    mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const formData = new FormData();
                        formData.append('file', audioBlob, 'recorded_audio.wav');
                        fetch('/process_audio', {
                            method: 'POST',
                            body: formData
                        }).then(response => response.json())
                          .then(data => {
                              document.getElementById("original_text").textContent = data.original_text;
                              document.getElementById("translated_text").textContent = data.translated_text;
                              const audio = new Audio(data.audio_url);
                              audio.play();
                          });
                        audioChunks = [];
                    };
                    mediaRecorder.start();
                });
            }

            function toggleRecording() {
                if (mediaRecorder && mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                    document.getElementById("recordButton").innerText = "Start Recording";
                } else {
                    startRecording();
                    document.getElementById("recordButton").innerText = "Stop Recording";
                }
            }
        </script>
    </head>
    <body>
        <h1>Audio Translator</h1>
        <button id="recordButton" onclick="toggleRecording()">Start Recording</button>
        <div style="display: flex; margin-top: 20px;">
            <div style="flex: 1; padding: 10px;">
                <h3>Original Text</h3>
                <p id="original_text">Recording will appear here...</p>
            </div>
            <div style="flex: 1; padding: 10px;">
                <h3>Translated Text</h3>
                <p id="translated_text">Translated text will appear here...</p>
            </div>
        </div>
    </body>
    </html>
    '''
    return render_template_string(HTML_TEMPLATE)

# Process audio route
@app.route('/process_audio', methods=['POST'])
def process_audio():
    logging.info("Received request to process audio")
    
    file = request.files['file']
    if file:
        # Create a temporary directory and save the audio file within it
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'recorded_audio.wav')
            file.save(file_path)
            logging.info(f"Audio file saved temporarily at {file_path}")

            # Upload audio to Google Cloud Storage
            try:
                bucket = storage_client.bucket(BUCKET_NAME)
                blob = bucket.blob('recorded_audio.wav')
                blob.upload_from_filename(file_path)
                logging.info(f"Audio file uploaded to Google Cloud Storage at gs://{BUCKET_NAME}/recorded_audio.wav")
            except Exception as e:
                logging.error(f"Failed to upload audio file to Google Cloud Storage: {e}")
                return jsonify({'error': 'Failed to upload audio to cloud storage.'}), 500

            # Transcribe with specified language code (using 'en-US')
            try:
                audio = speech.RecognitionAudio(uri=f'gs://{BUCKET_NAME}/recorded_audio.wav')
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    language_code='en-US',  # Specified language code
                    enable_automatic_punctuation=True
                )
                response = speech_client.recognize(config=config, audio=audio)
                original_text = ' '.join([result.alternatives[0].transcript for result in response.results])
                logging.info(f"Transcription complete: {original_text}")
                logging.info(f"Transcription output: {original_text}")
            except Exception as e:
                logging.error(f"Failed to transcribe audio: {e}")
                return jsonify({'error': 'Failed to transcribe audio.'}), 500

            # Translate text to Spanish
            try:
                translation = translate_client.translate(original_text, target_language='es')
                translated_text = translation['translatedText']
                logging.info(f"Translation complete: {translated_text}")
                logging.info(f"Translation output: {translated_text}")
            except Exception as e:
                logging.error(f"Failed to translate text: {e}")
                return jsonify({'error': 'Failed to translate text.'}), 500

            # Synthesize translated text to speech
            try:
                synthesis_input = texttospeech.SynthesisInput(text=translated_text)
                voice = texttospeech.VoiceSelectionParams(
                    language_code="es",
                    ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                )
                audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
                tts_response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
                logging.info("Text-to-speech synthesis complete")
            except Exception as e:
                logging.error(f"Failed to synthesize speech: {e}")
                return jsonify({'error': 'Failed to synthesize speech.'}), 500

            # Save the TTS audio to a BytesIO object and upload to Cloud Storage
            try:
                audio_output = io.BytesIO(tts_response.audio_content)
                audio_output.seek(0)
                translated_audio_blob = bucket.blob('translated_speech.mp3')
                translated_audio_blob.upload_from_file(audio_output, content_type='audio/mpeg')
                audio_url = translated_audio_blob.public_url
                logging.info(f"Synthesized audio uploaded to Google Cloud Storage at {audio_url}")
            except Exception as e:
                logging.error(f"Failed to upload synthesized audio to Google Cloud Storage: {e}")
                return jsonify({'error': 'Failed to upload synthesized audio.'}), 500

            # Return original and translated text, and a URL to the audio file
            return jsonify({
                'original_text': original_text,
                'translated_text': translated_text,
                'audio_url': audio_url
            })

    logging.error("No audio file found in request.")
    return jsonify({'error': 'File not found or invalid.'}), 400

if __name__ == '__main__':
    app.run(debug=True)



# from flask import Flask, request, render_template_string, send_file
# from google.cloud import storage, speech, texttospeech, translate_v2 as translate
# import os
# import requests
# import io

# app = Flask(__name__)

# # Set the environment variable for Google Cloud credentials
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\andre\\Downloads\\language-440220-e0110f2acfe7.json'

# # Initialize Google Cloud clients
# storage_client = storage.Client()
# speech_client = speech.SpeechClient()
# tts_client = texttospeech.TextToSpeechClient()
# translate_client = translate.Client()

# # Your bucket name
# BUCKET_NAME = 'portable-lang'

# # Home route
# @app.route('/', methods=['GET'])
# def home():
#     return "Welcome to the Home Page!"

# # Transcribe audio function
# def transcribe_audio(file_name):
#     audio = speech.RecognitionAudio(uri=f'gs://{BUCKET_NAME}/{file_name}')
#     config = speech.RecognitionConfig(
#         encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#         sample_rate_hertz=48000,
#         language_code='en-US',
#     )
#     response = speech_client.recognize(config=config, audio=audio)
#     transcriptions = [result.alternatives[0].transcript for result in response.results]
#     return ' '.join(transcriptions)

# # Synthesize speech function (Text-to-Speech)
# def synthesize_speech(text):
#     synthesis_input = texttospeech.SynthesisInput(text=text)
#     voice = texttospeech.VoiceSelectionParams(
#         language_code="es",  # Change to Spanish for TTS output
#         ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
#     )
#     audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
#     response = tts_client.synthesize_speech(
#         input=synthesis_input,
#         voice=voice,
#         audio_config=audio_config
#     )
#     audio_output = io.BytesIO(response.audio_content)
#     audio_output.seek(0)  # Reset the buffer pointer
#     return audio_output

# # Upload audio file route (for transcription and synthesis)
# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             bucket = storage_client.get_bucket(BUCKET_NAME)
#             blob = bucket.blob(file.filename)
#             blob.upload_from_file(file)
            
#             transcription = transcribe_audio(blob.name)
#             audio_output = synthesize_speech(transcription)
#             return send_file(audio_output, mimetype='audio/mpeg', as_attachment=True, download_name='output.mp3')

#     return '''
#     <h1>Upload File</h1>
#     <form method="post" enctype="multipart/form-data">
#         <input type="file" name="file" required>
#         <input type="submit" value="Upload">
#     </form>
#     '''

# # Upload text file route (for Text-to-Speech)
# @app.route('/upload_text', methods=['GET', 'POST'])
# def upload_text_file():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file and file.filename.endswith('.txt'):
#             # Upload the text file to Google Cloud Storage
#             bucket = storage_client.get_bucket(BUCKET_NAME)
#             blob = bucket.blob(file.filename)
#             blob.upload_from_file(file)

#             # Read the text content from the file
#             text_content = blob.download_as_text()

#             # Synthesize the text content into speech
#             audio_output = synthesize_speech(text_content)

#             # Return the synthesized audio file for download
#             return send_file(audio_output, mimetype='audio/mpeg', as_attachment=True, download_name='text_to_speech_output.mp3')

#     return '''
#     <h1>Upload Text File for Speech Synthesis</h1>
#     <form method="post" enctype="multipart/form-data">
#         <input type="file" name="file" accept=".txt" required>
#         <input type="submit" value="Upload and Convert to Speech">
#     </form>
#     '''

# # Upload combined route (audio upload, transcription, translation, and synthesis)
# @app.route('/upload_combined', methods=['GET', 'POST'])
# def upload_combined():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             bucket = storage_client.get_bucket(BUCKET_NAME)
#             blob = bucket.blob(file.filename)
#             blob.upload_from_file(file)

#             # Transcribe the audio file
#             transcription = transcribe_audio(blob.name)

#             # Translate the transcribed text to Spanish
#             translation = translate_client.translate(transcription, target_language='es')
#             translated_text = translation['translatedText']

#             # Synthesize the translated Spanish text into speech
#             audio_output = synthesize_speech(translated_text)

#             # Return the synthesized audio file for download
#             return send_file(audio_output, mimetype='audio/mpeg', as_attachment=True, download_name='translated_speech.mp3')

#     return '''
#     <h1>Upload Audio File for Combined Functionality</h1>
#     <form method="post" enctype="multipart/form-data">
#         <input type="file" name="file" required>
#         <input type="submit" value="Upload and Process">
#     </form>
#     '''

# # API Testing route (testing audio file upload)
# @app.route('/test_upload', methods=['POST'])
# def test_upload():
#     test_file_path = 'C:\\Users\\Owner\\Documents\\Google Cloud Testing\\Uploads\\You want to play Let.wav'
#     with open(test_file_path, 'rb') as test_file:
#         files = {'file': test_file}
#         response = requests.post(f'http://127.0.0.1:5000/upload_combined', files=files)
#     return f'Test Upload Response: {response.text}'

# if __name__ == '__main__':
#     app.run(debug=True)
