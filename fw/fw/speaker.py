# import pygame

# # Initialize pygame mixer
# pygame.mixer.init()

# # Load and play an audio file
# pygame.mixer.music.load("translated_audio.wav")  # Replace with your audio file path
# pygame.mixer.music.play()

# # Wait for the music to finish playing
# while pygame.mixer.music.get_busy():
#     continue


import pygame

class Speaker:
    def __init__(self):
        pygame.mixer.init()
    
    def play_audio(self, file_path):
        try:
            print(f"Loading audio file: {file_path}")
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            print("Playing audio...")
            
            # Wait for the music to finish playing
            while pygame.mixer.music.get_busy():
                continue
            
            print("Audio playback finished.")
        except pygame.error as e:
            print(f"Error playing audio: {e}")
    
    def stop_audio(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            print("Audio playback stopped.")


if __name__ == "__main__":
    speaker = Speaker()
    speaker.play_audio("translated_audio.wav")  

