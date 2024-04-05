import streamlit as st
import openai
from io import BytesIO
from pydub import AudioSegment
from pydub.utils import make_chunks

# Initialize OpenAI client
openai.api_key = st.secrets["OPENAI_API_KEY"]

def transcribe_audio_chunks(chunks):
    full_transcript = ""
    for chunk in chunks:
        try:
            # Convert chunk to bytes
            chunk_byte = BytesIO()
            chunk.export(chunk_byte, format="mp3")
            chunk_byte.seek(0)

            response = openai.Audio.transcribe(
                model="whisper-1", 
                file=chunk_byte
            )
            full_transcript += response['text'] + " "
        except Exception as e:
            st.error("Error in transcribing chunk: " + str(e))
    return full_transcript

def main():
    st.title("Speech to Text Conversion")
    st.write("Upload your audio file and convert it to text using OpenAI's Whisper API.")

    audio_file = st.file_uploader("Upload Audio", type=['mp3','m4a'])
    
    if audio_file is not None:
        with st.spinner('Transcribing...'):
            # Read and split audio file into chunks
            audio = AudioSegment.from_mp3(audio_file)
            chunks = make_chunks(audio, 300000)  # 24 seconds chunks

            transcript = transcribe_audio_chunks(chunks)
            st.write("Transcription:")
            st.text_area("Transcript", transcript, height=250)

if __name__ == "__main__":
    main()
