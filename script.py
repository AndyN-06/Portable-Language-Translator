import boto3
import os

# Initialize AWS Polly client
polly_client = boto3.client('polly', region_name='us-east-1')

def text_to_speech(text, file_name='speech.mp3'):
    response = polly_client.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Joanna'  # You can use any Polly voice here
    )
    
    # Save the audio stream as an MP3 file
    with open(file_name, 'wb') as file:
        file.write(response['AudioStream'].read())
    print(f'Audio saved as {file_name}')

# Example usage
text = "Hello, this is a test of AWS Polly."
text_to_speech(text)

# Optional: Play the audio file using an OS command
os.system(f"start {file_name}")  # This will play the file on Windows
