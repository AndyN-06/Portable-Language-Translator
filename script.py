# Importing the AWS SDK for Python (Boto3) to interact with AWS services.
import boto3  
import os

# Initialize AWS Polly client
# This creates a connection to AWS Polly in the specified region ('us-east-2'). Polly is used for text-to-speech services.
polly_client = boto3.client('polly', region_name='us-east-2')

# Function to convert text to speech and save it as an MP3 file
def text_to_speech(text, file_name='speech.mp3'):
    # Synthesize speech from the input text using AWS Polly
    response = polly_client.synthesize_speech(
        Text=text,          # The text that you want to convert to speech.
        OutputFormat='mp3', # Output format of the audio (MP3 in this case).
        VoiceId='Joanna'    # The voice ID to use (Polly supports multiple voices; 'Joanna' is a female US English voice).
    )
    
    # Save the audio stream as an MP3 file
    # This writes the audio stream (returned from AWS Polly) into a local MP3 file.
    with open(file_name, 'wb') as file:  # 'wb' mode opens the file for writing in binary mode.
        file.write(response['AudioStream'].read())  # Writing the audio stream to the file.
    
    print(f'Audio saved as {file_name}')  # Print a message indicating that the audio file has been saved.

# Example usage of the text_to_speech function
text = "Hello, this is a test of AWS Polly."  # The text you want to convert to speech.
text_to_speech(text)  # Call the function to convert the text to speech and save it as an MP3 file.

# Optional: Play the audio file using an OS command (this command is for Windows)
# The `start` command is used in Windows to open files with their default application.
os.system(f"start {'speech.mp3'}")  # This plays the saved audio file ('speech.mp3') using the default audio player on Windows.
