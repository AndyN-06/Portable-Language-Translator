import boto3
import time
import os
import requests

def transcribe_audio(mp3_file_path, job_name, bucket_name, region_name='us-east-1'):
    # Initialize the boto3 client for AWS Transcribe
    transcribe = boto3.client('transcribe', region_name=region_name)

    # Upload the MP3 file to an S3 bucket (must already exist)
    s3 = boto3.client('s3', region_name=region_name)
    s3_key = os.path.basename(mp3_file_path)
    s3.upload_file(mp3_file_path, bucket_name, s3_key)

    # Transcribe the audio file using AWS Transcribe
    audio_file_uri = f"s3://{bucket_name}/{s3_key}"
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': audio_file_uri},
        MediaFormat='mp3',
        LanguageCode='en-US'
    )

    # Wait for the transcription job to complete
    while True:
        response = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        status = response['TranscriptionJob']['TranscriptionJobStatus']
        if status == 'COMPLETED':
            print("Transcription completed.")
            break
        elif status == 'FAILED':
            print("Transcription failed.")
            return
        else:
            print(f"Transcription status: {status}. Waiting for completion...")
        time.sleep(10)

    # Get the transcription result URL
    transcript_file_uri = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
    
    # Fetch the transcription result from the URL
    transcript_response = requests.get(transcript_file_uri)
    transcript_text = transcript_response.json()['results']['transcripts'][0]['transcript']
    
    # Print the transcription result
    print(transcript_text)

if __name__ == "__main__":
    mp3_file = "speech.mp3"  # Path to the local MP3 file
    job_name = "test3"
    bucket_name = "translatetest03"

    transcribe_audio(mp3_file, job_name, bucket_name)
