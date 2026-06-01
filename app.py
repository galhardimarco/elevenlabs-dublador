import streamlit as st
import os
import subprocess
import numpy as np
import soundfile as sf
import pysrt
import requests
import time

st.set_page_config(page_title="ElevenLabs SRT Voice Generator", page_icon="🎙️", layout="wide")

# ===================== SEGURANÇA (igual ao seu) =====================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        st.title("🔐 Restricted Access")
        st.markdown("**This tool is private.**")
        
        password = st.text_input("Enter password to continue:", type="password")
        
        if st.button("Login", type="primary"):
            if password == st.secrets.get("APP_PASSWORD", "default_password"):
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("❌ Incorrect password!")
        return False
    return True

if not check_password():
    st.stop()

# ===================== FUNÇÕES =====================
def generate_tts(text, output_path, voice_id, stability, similarity):
    data = {
        "text": text,
        "model_id": "eleven_v3",
        "voice_settings": {"stability": stability, "similarity_boost": similarity}
    }
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    response = requests.post(url, headers=HEADERS, json=data)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    else:
        st.error(f"ElevenLabs Error: {response.status_code}")
        return False

def apply_ffmpeg_speed(input_path, output_path, speed):
    speed = max(0.5, min(speed, 100))
    filters = []
    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        filters.append("atempo=0.5")
        speed *= 2.0
    filters.append(f"atempo={speed:.5f}")
    filters.append("volume=1.2")
    command = ["ffmpeg", "-y", "-i", input_path, "-filter:a", ",".join(filters), "-ac", "2", "-vn", output_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def normalize_audio(data):
    peak = np.max(np.abs(data))
    return data / peak * 0.95 if peak > 0 else data
    
# ===================== CONFIG =====================
API_KEY = st.secrets["ELEVENLABS_API_KEY"]
HEADERS = {"xi-api-key": API_KEY, "Content-Type": "application/json"}

# ===================== MAPA DE PAÍSES =====================
COUNTRY_VOICES = {
    "Brazil": {"name": "Brazil Estive New", "id": "YU8EsJtXFMyKMxYtheDk"},
    "Korea": {"name": "Korean Young Gyu", "id": "h5eZa8VFAq0EQ8E81dfL"},
    "Japanese": {"name": "Japanese Otani", "id": "3JDquces8E8bkmvbh6Bc"},
    "Filipino": {"name": "Filipino Pocholo Gonzales", "id": "VCGhAh0uUPumdTXQNdzQ"},
}

# ===================== ESTADO =====================
if "selected_country" not in st.session_state:
    st.session_state.selected_country = None
if "selected_voice_name" not in st.session_state:
    st.session_state.selected_voice_name = None
if "selected_voice_id" not in st.session_state:
    st.session_state.selected_voice_id = None

# ===================== TELA 1: ESCOLHA DO PAÍS =====================
if st.session_state.selected_country is None:
    st.title("🌍 Welcome to ElevenLabs SRT Voice Generator")
    st.markdown("### Select your country to load the recommended voice")

    country = st.selectbox("Select your country", options=list(COUNTRY_VOICES.keys()))

    if st.button("Continue →", type="primary", use_container_width=True):
        st.session_state.selected_country = country
        st.session_state.selected_voice_name = COUNTRY_VOICES[country]["name"]
        st.session_state.selected_voice_id = COUNTRY_VOICES[country]["id"]
        st.rerun()

else:
    # ===================== TELA 2: GERAÇÃO (quase igual ao original) =====================
    st.title("🎙️ ElevenLabs SRT Voice Generator")
    st.caption(f"Country: **{st.session_state.selected_country}** • Voice: **{st.session_state.selected_voice_name}**")

    if st.button("← Change Country"):
        st.session_state.selected_country = None
        st.rerun()

    col1, col2 = st.columns([3, 2])

    with col1:
        uploaded_file = st.file_uploader("Upload .srt file", type=["srt"])

    with col2:
        st.selectbox("Voice", [st.session_state.selected_voice_name], disabled=True)

    st.subheader("⚙️ Voice Settings")
    col_a, col_b = st.columns(2)
    with col_a:
        stability = st.slider("Stability", 0.0, 1.0, 0.6, 0.05)
    with col_b:
        similarity = st.slider("Similarity Boost", 0.0, 1.0, 0.85, 0.05)

    if uploaded_file and st.button("🚀 Generate Audio", type="primary", use_container_width=True):
        task_id = f"task_{int(time.time())}"
        progress_bar = st.progress(0, text="Starting processing...")
        status_text = st.empty()

        try:
            temp_srt = f"temp_{task_id}.srt"
            with open(temp_srt, "wb") as f:
                f.write(uploaded_file.getbuffer())

            subtitles = pysrt.open(temp_srt, encoding='utf-8')
            total = len([s for s in subtitles if s.text.strip()])

            final_audio = np.zeros((0, 2), dtype=np.float32)
            sr = 44100
            VOICE_ID = st.session_state.selected_voice_id

            for i, sub in enumerate(subtitles):
                text = sub.text.strip().replace("\n", " ")
                if not text: 
                    continue

                status_text.text(f"🎤 Processing {i+1}/{total}: {text[:60]}...")

                temp_audio = f"seg_{task_id}_{i}.mp3"
                success = generate_tts(text, temp_audio, VOICE_ID, stability, similarity)
                if not success: 
                    continue

                audio_data, sr = sf.read(temp_audio)
                if audio_data.ndim == 1:
                    audio_data = np.column_stack([audio_data, audio_data])
                audio_data = normalize_audio(audio_data)

                # === AJUSTE DE VELOCIDADE ===
                start_ms = int(sub.start.ordinal)
                end_ms = int(sub.end.ordinal)
                available = end_ms - start_ms
                original_dur = int(len(audio_data) / sr * 1000)

                if original_dur > available * 1.05:
                    speed = min(original_dur / available, 2.0)
                    sped_path = f"sped_{task_id}_{i}.mp3"
                    apply_ffmpeg_speed(temp_audio, sped_path, speed)
                    
                    audio_data, sr = sf.read(sped_path)
                    if audio_data.ndim == 1:
                        audio_data = np.column_stack([audio_data, audio_data])
                    audio_data = normalize_audio(audio_data)
                    os.remove(sped_path)

                    if speed > 1.5:
                        st.warning(f"⚠️ **ALERT**: Segment {i+1} sped up **{speed:.2f}x**")
                    else:
                        st.info(f"⚡ Segment {i+1} sped up **{speed:.2f}x**")

                start_sample = int(start_ms / 1000.0 * sr)
                end_sample = start_sample + len(audio_data)

                if end_sample > len(final_audio):
                    new_audio = np.zeros((end_sample, 2), dtype=np.float32)
                    new_audio[:len(final_audio)] = final_audio
                    final_audio = new_audio

                final_audio[start_sample:end_sample] = audio_data
                progress_bar.progress((i + 1) / total)

                if os.path.exists(temp_audio):
                    os.remove(temp_audio)

            output_path = f"final_{task_id}.mp3"
            sf.write(output_path, final_audio, sr)

            st.success("✅ Audio generated successfully!")

            with open(output_path, "rb") as f:
                st.download_button(
                    "📥 Download Final Audio",
                    data=f,
                    file_name=uploaded_file.name.replace(".srt", f"_dubbed_{st.session_state.selected_voice_name}.mp3"),
                    mime="audio/mpeg",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            for f in os.listdir():
                if task_id in f:
                    try: os.remove(f)
                    except: pass

