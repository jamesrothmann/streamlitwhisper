import streamlit as st
import openai
from io import BytesIO

# Initialize OpenAI client
openai.api_key = st.secrets["OPENAI_API_KEY"]

def transcribe_audio(audio_file):
    try:
        # Convert the uploaded file to bytes
        audio_bytes = audio_file.getvalue()
        response = openai.Audio.transcribe(
            model="whisper-1", 
            file=BytesIO(audio_bytes)
        )
        return response['text']
    except Exception as e:
        return str(e)

def main():
    st.title("Speech to Text Conversion")
    st.write("Upload your audio file and convert it to text using OpenAI's Whisper API.")

    audio_file = st.file_uploader("Upload Audio", type=['mp3', 'wav', 'mp4', 'mpeg', 'mpga', 'm4a', 'webm'])
    
    if audio_file is not None:
        with st.spinner('Transcribing...'):
            transcript = transcribe_audio(audio_file)
            st.write("Transcription:")
            st.text_area("Transcript", transcript, height=250)

if __name__ == "__main__":
    main()
