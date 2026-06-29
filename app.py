import streamlit as st
import os
import subprocess
import numpy as np
import soundfile as sf
import pysrt
import requests
import time

st.set_page_config(
    page_title="ElevenLabs SRT Voice Generator",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CONFIG (moved to top for clarity) =====================
API_KEY = st.secrets["ELEVENLABS_API_KEY"]
HEADERS = {"xi-api-key": API_KEY, "Content-Type": "application/json"}

# ===================== MAPA DE VOZES =====================
COUNTRY_VOICES = {
    "Brazil 🇧🇷": {"name": "Brazil Estive New", "id": "YU8EsJtXFMyKMxYtheDk"},
    "Korea 🇰🇷": {"name": "Korean Young Gyu", "id": "h5eZa8VFAq0EQ8E81dfL"},
    "Japanese 🇯🇵": {"name": "Japanese Otani", "id": "3JDquces8E8bkmvbh6Bc"},
    "Filipino 🇵🇭": {"name": "Filipino Pocholo Gonzales", "id": "VCGhAh0uUPumdTXQNdzQ"},
}

# ===================== FUNÇÕES =====================
def generate_tts(text, output_path, voice_id, stability, similarity, model_id="eleven_v3"):
    """Gera áudio TTS via ElevenLabs"""
    data = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity
        }
    }
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    try:
        response = requests.post(url, headers=HEADERS, json=data, timeout=60)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            st.error(f"ElevenLabs Error {response.status_code}: {response.text[:200]}")
            return False
    except Exception as e:
        st.error(f"Request failed: {e}")
        return False


def apply_ffmpeg_speed(input_path, output_path, speed):
    """Acelera áudio mantendo tom (pitch)"""
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

    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter:a", ",".join(filters),
        "-ac", "2", "-vn", output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def normalize_audio(data):
    """Normaliza pico de áudio para -0.5 dBFS approx"""
    peak = np.max(np.abs(data))
    return data / peak * 0.95 if peak > 0 else data


@st.cache_data(ttl=300, show_spinner=False)
def get_subscription_info():
    """Busca uso de caracteres da conta ElevenLabs"""
    try:
        resp = requests.get(
            "https://api.elevenlabs.io/v1/user/subscription",
            headers=HEADERS,
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            used = data.get("character_count", 0)
            limit = data.get("character_limit", 0)
            tier = data.get("tier", "Unknown")
            remaining = max(0, limit - used) if limit > 0 else 0
            pct = round((used / limit * 100), 1) if limit > 0 else 0
            return {
                "tier": tier,
                "used": used,
                "limit": limit,
                "remaining": remaining,
                "pct": pct,
                "error": None
            }
        else:
            return {"error": f"API returned {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# ===================== SEGURANÇA =====================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        st.title("🔐 Restricted Access")
        st.markdown("**This tool is private.** Please enter the password to continue.")
        
        password = st.text_input("Password", type="password", key="pwd_input")
        
        if st.button("Login", type="primary", use_container_width=True):
            if password == st.secrets.get("APP_PASSWORD", "default_password"):
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("❌ Incorrect password. Please try again.")
        st.stop()
    return True


if not check_password():
    st.stop()

# ===================== ESTADO DA SESSÃO =====================
if "selected_country" not in st.session_state:
    st.session_state.selected_country = None
if "selected_voice_name" not in st.session_state:
    st.session_state.selected_voice_name = None
if "selected_voice_id" not in st.session_state:
    st.session_state.selected_voice_id = None

# ===================== TELA 1: ESCOLHA DO PAÍS / VOZ =====================
if st.session_state.selected_country is None:
    st.title("🌍 ElevenLabs SRT Voice Generator")
    st.markdown("### Select your country / voice to get started")

    col1, col2 = st.columns([2, 1])
    with col1:
        country = st.selectbox(
            "Choose voice / language",
            options=list(COUNTRY_VOICES.keys()),
            index=0,
            help="Each option uses a high-quality ElevenLabs voice optimized for that language/region."
        )

    with col2:
        st.info("**Recommended for Brazilian Portuguese content:** Brazil 🇧🇷 voice")

    if st.button("Continue →", type="primary", use_container_width=True):
        st.session_state.selected_country = country
        st.session_state.selected_voice_name = COUNTRY_VOICES[country]["name"]
        st.session_state.selected_voice_id = COUNTRY_VOICES[country]["id"]
        st.rerun()

else:
    # ===================== TELA 2: GERAÇÃO PRINCIPAL =====================
    st.title("🎙️ ElevenLabs SRT Voice Generator")
    st.caption(f"**{st.session_state.selected_country}** • {st.session_state.selected_voice_name}")

    # Botão para trocar de voz/país
    if st.button("← Change Voice / Country", type="secondary"):
        st.session_state.selected_country = None
        st.rerun()

    # ===================== SIDEBAR: CRÉDITOS + CONFIG =====================
    with st.sidebar:
        st.header("📊 ElevenLabs Account")
        sub_info = get_subscription_info()
        
        if sub_info.get("error"):
            st.error(f"Could not load account info: {sub_info['error']}")
            if st.button("🔄 Retry"):
                st.cache_data.clear()
                st.rerun()
        else:
            st.metric("Current Plan", sub_info["tier"])
            
            cols = st.columns(2)
            cols[0].metric("Characters Used", f"{sub_info['used']:,}")
            cols[1].metric("Remaining", f"{sub_info['remaining']:,}")
            
            st.progress(sub_info["pct"] / 100.0, text=f"{sub_info['pct']}% of monthly quota used")
            
            if sub_info["pct"] > 85:
                st.error("⚠️ **Quota almost exhausted!** Consider upgrading or waiting for reset.")
            elif sub_info["pct"] > 70:
                st.warning("⚠️ Quota running low. Monitor usage.")
            else:
                st.success("✅ Good quota balance")

            st.caption("Data refreshes every 5 minutes. Use 'Rerun' button above to force refresh.")

        st.divider()
        st.header("⚙️ Generation Settings")
        
        force_fit = st.checkbox(
            "Force segments to fit timing (truncate if needed)",
            value=False,
            help="Only enable this if you want strict subtitle timing. When enabled and even 2x speed is not enough, the end of the sentence will be cut off. For Bible teaching, it's often better to leave it unchecked and adjust SRT timing manually if needed."
        )
        
        model_id = st.selectbox(
            "ElevenLabs Model",
            options=["eleven_v3", "eleven_turbo_v2_5"],
            index=0,
            help="eleven_v3 = highest quality (slower, more expensive). eleven_turbo_v2_5 = faster & cheaper for drafts."
        )

    # ===================== ÁREA PRINCIPAL =====================
    col_upload, col_info = st.columns([3, 2])

    with col_upload:
        uploaded_file = st.file_uploader(
            "📁 Upload your .srt file",
            type=["srt"],
            help="Upload the subtitle file exported from your video editor or transcription tool."
        )

    with col_info:
        st.info("**Tip:** For best results with Bible teaching videos, use **Stability 0.55-0.65** and **Similarity 0.80-0.90** for consistent, trustworthy narration voice.")

    # Configurações de voz (sliders)
    st.subheader("🎛️ Voice Quality Settings")
    col_stab, col_sim = st.columns(2)
    
    with col_stab:
        stability = st.slider(
            "Stability",
            min_value=0.0, max_value=1.0, value=0.60, step=0.05,
            help="Higher = more consistent voice (recommended for teaching). Lower = more expressive/emotional."
        )
    with col_sim:
        similarity = st.slider(
            "Similarity Boost",
            min_value=0.0, max_value=1.0, value=0.85, step=0.05,
            help="Higher = closer to the original voice timbre. Good for brand consistency."
        )

    # ===================== PRÉVIA E GERAÇÃO =====================
    if uploaded_file:
        # Ler SRT para mostrar preview e calcular caracteres
        try:
            temp_srt_path = f"temp_preview_{int(time.time())}.srt"
            with open(temp_srt_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            subtitles = pysrt.open(temp_srt_path, encoding='utf-8')
            valid_subs = [s for s in subtitles if s.text.strip()]
            total_chars = sum(len(s.text.strip()) for s in valid_subs)
            total_segments = len(valid_subs)
            
            # Preview
            with st.expander("📋 SRT Preview & Stats", expanded=False):
                st.write(f"**Total segments:** {total_segments}")
                st.write(f"**Total characters:** {total_chars:,}")
                if total_chars > 0:
                    est_cost = total_chars / 1000 * 0.10
                    st.write(f"**Estimated cost (eleven_v3):** ~${est_cost:.2f} USD")
                st.caption("First 5 segments:")
                for i, sub in enumerate(valid_subs[:5]):
                    st.text(f"[{i+1}] {sub.text[:80]}{'...' if len(sub.text) > 80 else ''}")
            
            os.remove(temp_srt_path)

        except Exception as e:
            st.error(f"Error reading SRT: {e}")
            st.stop()

        # Botão de gerar
        if st.button("🚀 Generate Dubbed Audio", type="primary", use_container_width=True):
            task_id = f"task_{int(time.time())}"
            progress_bar = st.progress(0, text="Starting processing...")
            status_text = st.empty()

            try:
                # Salvar SRT temporário
                temp_srt = f"temp_{task_id}.srt"
                with open(temp_srt, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                subtitles = pysrt.open(temp_srt, encoding='utf-8')
                valid_subs = [s for s in subtitles if s.text.strip()]
                total = len(valid_subs)

                final_audio = np.zeros((0, 2), dtype=np.float32)
                sr = 44100
                VOICE_ID = st.session_state.selected_voice_id

                # === ESTATÍSTICAS PARA O RESUMO FINAL ===
                sped_up_segments = []      # lista de (número, velocidade)
                truncated_segments = []    # lista de (número, ms_cortados)

                for idx, sub in enumerate(valid_subs):
                    text = sub.text.strip().replace("\n", " ")
                    if not text:
                        continue

                    progress = (idx + 1) / total
                    status_text.text(f"🎤 Processing segment {idx+1}/{total} — {text[:55]}...")
                    progress_bar.progress(progress)

                    temp_audio = f"seg_{task_id}_{idx}.mp3"
                    success = generate_tts(text, temp_audio, VOICE_ID, stability, similarity, model_id)
                    if not success:
                        continue

                    # Carregar áudio gerado
                    audio_data, sr = sf.read(temp_audio)
                    if audio_data.ndim == 1:
                        audio_data = np.column_stack([audio_data, audio_data])
                    audio_data = normalize_audio(audio_data)

                    # === LÓGICA DE TIMING ===
                    start_ms = int(sub.start.ordinal)
                    end_ms = int(sub.end.ordinal)
                    available = end_ms - start_ms
                    original_dur = int(len(audio_data) / sr * 1000)

                    do_truncate = False
                    speed_used = 1.0
                    truncated_amount_ms = 0

                    if original_dur > available * 1.05:
                        speed = original_dur / available
                        speed_used = min(speed, 2.0)

                        # Aplicar aceleração
                        sped_path = f"sped_{task_id}_{idx}.mp3"
                        apply_ffmpeg_speed(temp_audio, sped_path, speed_used)
                        
                        audio_data, sr = sf.read(sped_path)
                        if audio_data.ndim == 1:
                            audio_data = np.column_stack([audio_data, audio_data])
                        audio_data = normalize_audio(audio_data)
                        os.remove(sped_path)

                        new_dur = int(len(audio_data) / sr * 1000)

                        # Truncar apenas se o usuário ativou "force_fit" E ainda não coube
                        if force_fit and new_dur > available:
                            max_samples = int(available / 1000.0 * sr)
                            truncated_amount_ms = new_dur - available
                            audio_data = audio_data[:max_samples]
                            do_truncate = True
                            truncated_segments.append((idx + 1, truncated_amount_ms))

                        # Registrar para o resumo final (sem mostrar mensagens ao vivo para não poluir a tela)
                        if do_truncate and truncated_amount_ms > 250:
                            truncated_segments.append((idx + 1, truncated_amount_ms))
                        elif speed_used > 1.5:
                            sped_up_segments.append((idx + 1, speed_used))
                        elif speed_used > 1.25:
                            sped_up_segments.append((idx + 1, speed_used))

                    # Inserir no áudio final no tempo correto da legenda
                    start_sample = int(start_ms / 1000.0 * sr)
                    end_sample = start_sample + len(audio_data)

                    # Garantir que o buffer final seja grande o suficiente
                    if end_sample > len(final_audio):
                        new_len = end_sample + 1000  # margem
                        new_audio = np.zeros((new_len, 2), dtype=np.float32)
                        new_audio[:len(final_audio)] = final_audio
                        final_audio = new_audio

                    final_audio[start_sample:end_sample] = audio_data

                    # Limpar temporários do segmento
                    if os.path.exists(temp_audio):
                        os.remove(temp_audio)

                # Salvar arquivo final
                output_path = f"final_{task_id}.mp3"
                sf.write(output_path, final_audio, sr)

                progress_bar.progress(1.0)
                status_text.empty()
                st.success("✅ Audio generated successfully!")

                # ===================== RESUMO FINAL =====================
                with st.container(border=True):
                    st.subheader("📋 Processing Summary")

                    total_sped = len(sped_up_segments)
                    total_truncated = len(truncated_segments)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Segments", total)
                    col2.metric("Needed Speed-up (>1.25x)", total_sped)
                    col3.metric("Truncated", total_truncated)

                    if total_truncated > 0 or total_sped > 0:
                        st.markdown("**Segments that needed attention:**")

                        # Mostrar segmentos truncados
                        if total_truncated > 0:
                            trunc_text = ", ".join([f"**{num}** (cut {ms}ms)" for num, ms in truncated_segments])
                            st.markdown(f"- Truncated: {trunc_text}")

                        # Mostrar todos os segmentos que precisaram de aceleração
                        if total_sped > 0:
                            sped_text = ", ".join([f"**{num}** ({spd:.2f}x)" for num, spd in sped_up_segments])
                            st.markdown(f"- Sped up: {sped_text}")

                        st.caption("💡 Tip: For important teaching content, consider adjusting the original SRT timing instead of relying heavily on speed-up or truncation.")
                    else:
                        st.success("All segments fit well with minimal or no speed adjustment. Great job on the SRT timing!")

                # Botão de download (bem visível depois do resumo)
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Final Dubbed Audio (.mp3)",
                        data=f,
                        file_name=uploaded_file.name.replace(".srt", f"_dubbed_{st.session_state.selected_voice_name.replace(' ', '_')}.mp3"),
                        mime="audio/mpeg",
                        use_container_width=True,
                        type="primary"
                    )

                st.balloons()

            except Exception as e:
                st.error(f"❌ Unexpected error during processing: {e}")
                st.exception(e)
            finally:
                # Limpeza robusta de arquivos temporários
                for f_name in os.listdir():
                    if f_name.startswith(("temp_", "seg_", "sped_", "final_")) and task_id in f_name:
                        try:
                            os.remove(f_name)
                        except:
                            pass

    else:
        st.info("👆 Upload an SRT file above to begin generation.")

    # Rodapé / instruções
    st.divider()
    with st.expander("ℹ️ How to use this tool (for non-technical users)"):
        st.markdown("""
        1. **Select your voice** (country) on the first screen.
        2. **Upload** the .srt file from your video project.
        3. Adjust **Stability** and **Similarity** if needed (defaults are good for teaching videos).
        4. Check **"Force segments to fit timing"** (recommended for slide sync).
        5. Click **Generate**.
        6. Download the final .mp3 and import it into your video editor aligned with the subtitles.

        **Tip for Bible videos:** Keep Stability around 0.55-0.65 and Similarity 0.80-0.90 so the voice sounds calm, consistent and trustworthy.
        """)

    st.caption("Made with ❤️ for faithful content creators • Powered by ElevenLabs + Streamlit")
