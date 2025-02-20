from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import sys
from translator_device import TranslatorDevice  # Adjust the import path as needed

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Create an instance of TranslatorDevice
translator_device = TranslatorDevice()

@app.route('/set_settings', methods=['POST'])
def set_settings():
    data = request.get_json()
    base_language = data.get('baseLanguage')
    gender = data.get('gender')

    if not base_language or not gender:
        return jsonify({'status': 'error', 'message': 'Invalid settings.'}), 400

    translator_device.set_settings(base_language, gender)
    return jsonify({'status': 'success', 'message': 'Settings updated.'}), 200

if __name__ == "__main__":
    translator_thread = threading.Thread(target=translator_device.start)
    translator_thread.daemon = True  # Daemonize thread
    translator_thread.start()

    try:
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit()