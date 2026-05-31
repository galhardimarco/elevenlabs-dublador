import streamlit as st
import os
import subprocess
import numpy as np
import soundfile as sf
import pysrt
import requests
import time

st.set_page_config(page_title="ElevenLabs Dublador", page_icon="🎙️", layout="wide")

# ===================== SEGURANÇA =====================
def check_password():
    """Proteção com senha"""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        st.title("🔐 Acesso Restrito")
        st.markdown("**Esta ferramenta é privada.**")
        
        password = st.text_input("Digite a senha para continuar:", type="password")
        
        if st.button("Entrar", type="primary"):
            if password == st.secrets.get("APP_PASSWORD", "senha_padrao"):
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("❌ Senha incorreta!")
        return False
    return True

# ===================== VERIFICAÇÃO DE SENHA =====================
if not check_password():
    st.stop()

# ===================== CONFIGURAÇÕES =====================
API_KEY = st.secrets["ELEVENLABS_API_KEY"]

HEADERS = {"xi-api-key": API_KEY, "Content-Type": "application/json"}

# ===================== VOZES =====================
VOICES = {
    "Korean Young Gyu": "h5eZa8VFAq0EQ8E81dfL",
    "Estive New Brazil": "YU8EsJtXFMyKMxYtheDk",
}

# ===================== FUNÇÕES =====================
def generate_tts(text, output_path, voice_id, stability, similarity):
    data = {
        "text": text,
        "model_id": "eleven_v3",
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity
        }
    }
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    response = requests.post(url, headers=HEADERS, json=data)
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    else:
        st.error(f"Erro ElevenLabs: {response.status_code}")
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

# ===================== INTERFACE =====================
st.title("🎙️ ElevenLabs SRT Dublador")
st.markdown("Transforme suas legendas em áudio com voz IA")

col1, col2 = st.columns([3, 2])

with col1:
    uploaded_file = st.file_uploader("Envie seu arquivo .srt", type=["srt"])

with col2:
    selected_voice_name = st.selectbox("Escolha a Voz", options=list(VOICES.keys()))
    VOICE_ID = VOICES[selected_voice_name]

st.subheader("⚙️ Ajustes da Voz")
col_a, col_b = st.columns(2)
with col_a:
    stability = st.slider("Estabilidade", 0.0, 1.0, 0.5, 0.05)
with col_b:
    similarity = st.slider("Similarity Boost", 0.0, 1.0, 0.75, 0.05)

if uploaded_file and st.button("🚀 Gerar Áudio", type="primary", use_container_width=True):
    task_id = f"task_{int(time.time())}"
    
    progress_bar = st.progress(0, text="Iniciando processamento...")
    status_text = st.empty()

    try:
        temp_srt = f"temp_{task_id}.srt"
        with open(temp_srt, "wb") as f:
            f.write(uploaded_file.getbuffer())

        subtitles = pysrt.open(temp_srt, encoding='utf-8')
        total = len([s for s in subtitles if s.text.strip()])

        final_audio = np.zeros((0, 2), dtype=np.float32)
        sr = 44100

        for i, sub in enumerate(subtitles):
            text = sub.text.strip().replace("\n", " ")
            if not text: 
                continue

            status_text.text(f"🎤 Processando {i+1}/{total}: {text[:60]}...")

            temp_audio = f"seg_{task_id}_{i}.mp3"
            success = generate_tts(text, temp_audio, VOICE_ID, stability, similarity)
            if not success: 
                continue

            audio_data, sr = sf.read(temp_audio)
            if audio_data.ndim == 1:
                audio_data = np.column_stack([audio_data, audio_data])
            audio_data = normalize_audio(audio_data)

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

        st.success("✅ Áudio gerado com sucesso!")

        with open(output_path, "rb") as f:
            st.download_button(
                "📥 Baixar Áudio Final",
                data=f,
                file_name=uploaded_file.name.replace(".srt", f"_dublado_{selected_voice_name}.mp3"),
                mime="audio/mpeg",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Erro durante o processamento: {e}")
    finally:
        # Limpeza de arquivos temporários
        for f in os.listdir():
            if task_id in f:
                try: 
                    os.remove(f)
                except: 
                    pass
