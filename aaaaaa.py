from cartesia import Cartesia
import os

client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))

# Get all available voices
voices = client.voices.list()
print(voices[1]["id"])