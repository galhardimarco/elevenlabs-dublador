import streamlit as st
import os
import subprocess
import numpy as np
import soundfile as sf
import pysrt
import requests
import time
import queue
import uuid
from datetime import datetime

st.set_page_config(page_title="ElevenLabs SRT Voice Generator", page_icon="🎙️", layout="wide")

# ===================== CONFIG =====================
API_KEY = st.secrets["ELEVENLABS_API_KEY"]
HEADERS = {"xi-api-key": API_KEY, "Content-Type": "application/json"}

COUNTRY_VOICES = {
    "Brazil 🇧🇷": {"name": "Brazil Estive New", "id": "YU8EsJtXFMyKMxYtheDk"},
    "Korea 🇰🇷": {"name": "Korean Young Gyu", "id": "h5eZa8VFAq0EQ8E81dfL"},
    "Japanese 🇯🇵": {"name": "Japanese Otani", "id": "3JDquces8E8bkmvbh6Bc"},
    "Filipino 🇵🇭": {"name": "Filipino Pocholo Gonzales", "id": "VCGhAh0uUPumdTXQNdzQ"},
}

# ===================== QUEUE =====================
if "queue" not in st.session_state:
    st.session_state.queue = []
if "tasks" not in st.session_state:
    st.session_state.tasks = {}

def add_to_queue(task_id):
    if task_id not in st.session_state.queue:
        st.session_state.queue.append(task_id)

def get_position(task_id):
    try:
        return st.session_state.queue.index(task_id)
    except:
        return -1

def remove_from_queue(task_id):
    if task_id in st.session_state.queue:
        st.session_state.queue.remove(task_id)

# ===================== PROCESSING =====================
def generate_tts(text, output_path, voice_id, stability, similarity, model_id):
    data = {"text": text, "model_id": model_id, "voice_settings": {"stability": stability, "similarity_boost": similarity}}
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    resp = requests.post(url, headers=HEADERS, json=data, timeout=90)
    if resp.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(resp.content)
        return True
    raise Exception(f"ElevenLabs error {resp.status_code}")

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
    subprocess.run(["ffmpeg", "-y", "-i", input_path, "-filter:a", ",".join(filters), "-ac", "2", "-vn", output_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def normalize_audio(data):
    peak = np.max(np.abs(data))
    return data / peak * 0.95 if peak > 0 else data

def process_job(task_id):
    job = st.session_state.tasks[task_id]
    file_bytes = job["file_bytes"]
    voice_id = job["voice_id"]
    stability = job["stability"]
    similarity = job["similarity"]
    model_id = job["model_id"]
    force_fit = job["force_fit"]

    temp_srt = f"temp_{task_id}.srt"
    with open(temp_srt, "wb") as f:
        f.write(file_bytes)

    subtitles = pysrt.open(temp_srt, encoding='utf-8')
    valid_subs = [s for s in subtitles if s.text.strip()]
    total = len(valid_subs)

    final_audio = np.zeros((0, 2), dtype=np.float32)
    sr = 44100
    sped_up = []
    truncated = []

    progress = st.progress(0, text="Starting...")

    for idx, sub in enumerate(valid_subs):
        text = sub.text.strip().replace("\n", " ")
        if not text: continue

        progress.progress((idx + 1) / total, text=f"Processing segment {idx+1}/{total}...")

        temp_audio = f"seg_{task_id}_{idx}.mp3"
        generate_tts(text, temp_audio, voice_id, stability, similarity, model_id)

        audio_data, sr = sf.read(temp_audio)
        if audio_data.ndim == 1:
            audio_data = np.column_stack([audio_data, audio_data])
        audio_data = normalize_audio(audio_data)

        start_ms = int(sub.start.ordinal)
        end_ms = int(sub.end.ordinal)
        available = end_ms - start_ms
        original_dur = int(len(audio_data) / sr * 1000)

        speed_used = 1.0
        if original_dur > available * 1.05:
            speed = original_dur / available
            speed_used = min(speed, 2.0)

            sped_path = f"sped_{task_id}_{idx}.mp3"
            apply_ffmpeg_speed(temp_audio, sped_path, speed_used)
            audio_data, sr = sf.read(sped_path)
            if audio_data.ndim == 1:
                audio_data = np.column_stack([audio_data, audio_data])
            audio_data = normalize_audio(audio_data)
            os.remove(sped_path)

            new_dur = int(len(audio_data) / sr * 1000)
            if force_fit and new_dur > available + 50:
                max_samples = int(available / 1000.0 * sr)
                audio_data = audio_data[:max_samples]
                truncated.append((idx + 1, new_dur - available))
            elif speed_used > 1.25:
                sped_up.append((idx + 1, speed_used))

        start_sample = int(start_ms / 1000.0 * sr)
        end_sample = start_sample + len(audio_data)
        if end_sample > len(final_audio):
            new_audio = np.zeros((end_sample + 2000, 2), dtype=np.float32)
            new_audio[:len(final_audio)] = final_audio
            final_audio = new_audio
        final_audio[start_sample:end_sample] = audio_data

        if os.path.exists(temp_audio):
            os.remove(temp_audio)

    output_path = f"final_{task_id}.mp3"
    sf.write(output_path, final_audio, sr)
    os.remove(temp_srt)

    st.session_state.tasks[task_id]["status"] = "done"
    st.session_state.tasks[task_id]["result_path"] = output_path
    st.session_state.tasks[task_id]["filename"] = job["filename"].replace(".srt", "_dubbed.mp3")
    st.session_state.tasks[task_id]["sped_up"] = sped_up
    st.session_state.tasks[task_id]["truncated"] = truncated
    st.session_state.tasks[task_id]["total_segments"] = total

    remove_from_queue(task_id)

# ===================== SEGURANÇA =====================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if not st.session_state.password_correct:
        st.title("🔐 Restricted Access")
        pwd = st.text_input("Password", type="password")
        if st.button("Login", type="primary", use_container_width=True):
            if pwd == st.secrets.get("APP_PASSWORD", "default_password"):
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("❌ Incorrect password")
        st.stop()
    return True

if not check_password():
    st.stop()

# ===================== ESTADO =====================
if "my_task_id" not in st.session_state:
    st.session_state.my_task_id = None
if "selected_country" not in st.session_state:
    st.session_state.selected_country = None

# ===================== TELA 1: VOZ =====================
if st.session_state.selected_country is None:
    st.title("🌍 ElevenLabs SRT Voice Generator")
    st.markdown("### Select your voice / language")

    country = st.selectbox("Choose voice", options=list(COUNTRY_VOICES.keys()), index=0)
    st.info("**Recommended for Brazilian Portuguese content:** Brazil voice")

    if st.button("Continue →", type="primary", use_container_width=True):
        st.session_state.selected_country = country
        st.session_state.selected_voice_name = COUNTRY_VOICES[country]["name"]
        st.session_state.selected_voice_id = COUNTRY_VOICES[country]["id"]
        st.rerun()

else:
    # ===================== TELA 2 - UI COMO NA IMAGEM =====================
    st.title("🎙️ ElevenLabs SRT Voice Generator")
    st.caption(f"{st.session_state.selected_country} • {st.session_state.selected_voice_name}")

    if st.button("← Change Voice / Country"):
        st.session_state.selected_country = None
        st.rerun()

    # SIDEBAR
    with st.sidebar:
        st.header("ElevenLabs Account")
        try:
            resp = requests.get("https://api.elevenlabs.io/v1/user/subscription", headers=HEADERS, timeout=8)
            if resp.status_code == 200:
                d = resp.json()
                used = d.get("character_count", 0)
                limit = d.get("character_limit", 0)
                tier = d.get("tier", "creator")
                remaining = max(0, limit - used)
                pct = (used / limit * 100) if limit > 0 else 0

                st.write(f"**Current Plan**  \n{tier}")
                c1, c2 = st.columns(2)
                c1.metric("Characters Used", f"{used:,}")
                c2.metric("Remaining", f"{remaining:,}")
                st.progress(pct / 100, text=f"{pct:.1f}% of monthly quota used")

                if pct < 70:
                    st.success("Good quota balance")
                elif pct < 85:
                    st.warning("Quota running low")
                else:
                    st.error("Quota almost exhausted")
        except:
            pass

        st.divider()
        st.header("⚙️ Generation Settings")
        force_fit = st.checkbox("Force segments to fit timing (truncate if needed)", value=False)
        model_id = st.selectbox("ElevenLabs Model", ["eleven_v3", "eleven_turbo_v2_5"], index=0)

    # UPLOAD + TIP
    col_upload, col_tip = st.columns([3, 2])
    with col_upload:
        uploaded_file = st.file_uploader("📁 Upload your .srt file", type=["srt"])
    with col_tip:
        st.info("💡 Tip: For best results with Bible teaching videos, use **Stability 0.55-0.65** and **Similarity 0.80-0.90** for consistent, trustworthy narration voice.")

    # VOICE QUALITY SETTINGS
    st.subheader("🎛️ Voice Quality Settings")
    col1, col2 = st.columns(2)
    with col1:
        stability = st.slider("Stability", 0.0, 1.0, 0.60, 0.05, help="Higher = more consistent voice")
    with col2:
        similarity = st.slider("Similarity Boost", 0.0, 1.0, 0.85, 0.05, help="Higher = closer to original voice")

    # SRT PREVIEW (igual ao original)
    if uploaded_file:
        try:
            tmp = f"preview_{int(time.time())}.srt"
            with open(tmp, "wb") as f:
                f.write(uploaded_file.getbuffer())
            subs = pysrt.open(tmp, encoding='utf-8')
            valid = [s for s in subs if s.text.strip()]
            chars = sum(len(s.text.strip()) for s in valid)

            with st.expander("SRT Preview & Stats"):
                st.write(f"**Total segments:** {len(valid)}")
                st.write(f"**Total characters:** {chars:,}")
                st.write(f"**Estimated cost (eleven_v3):** ~${chars/1000 * 0.10:.2f} USD")
                st.write("**First 5 segments:**")
                for i, s in enumerate(valid[:5]):
                    st.write(f"[{i+1}] {s.text.strip()[:80]}{'...' if len(s.text.strip()) > 80 else ''}")
            os.remove(tmp)
        except:
            pass

    # ===================== GERAÇÃO =====================
    if st.session_state.my_task_id is None:
        if uploaded_file and st.button("Generate Dubbed Audio", type="primary", use_container_width=True):
            task_id = str(uuid.uuid4())[:8]
            st.session_state.tasks[task_id] = {
                "status": "queued",
                "file_bytes": uploaded_file.getbuffer().tobytes(),
                "filename": uploaded_file.name,
                "voice_id": st.session_state.selected_voice_id,
                "stability": stability,
                "similarity": similarity,
                "model_id": model_id,
                "force_fit": force_fit,
            }
            add_to_queue(task_id)
            st.session_state.my_task_id = task_id
            st.rerun()

    else:
        task_id = st.session_state.my_task_id
        task = st.session_state.tasks.get(task_id, {})
        pos = get_position(task_id)

        if task.get("status") == "queued":
            if pos == 0:
                st.warning("It's your turn! Processing now...")
                with st.spinner("Generating audio... Please wait."):
                    try:
                        process_job(task_id)
                        st.rerun()
                    except Exception as e:
                        st.session_state.tasks[task_id]["status"] = "error"
                        st.session_state.tasks[task_id]["error"] = str(e)
                        st.rerun()
            else:
                st.info(f"You are in position **#{pos + 1}** in the queue.")
                if st.button("Refresh status"):
                    st.rerun()

        elif task.get("status") == "done":
            st.success("Audio generated successfully!")

            result_path = task.get("result_path")
            if result_path and os.path.exists(result_path):
                with open(result_path, "rb") as f:
                    st.download_button(
                        "Download Final Dubbed Audio",
                        data=f,
                        file_name=task.get("filename", "dubbed.mp3"),
                        mime="audio/mpeg",
                        use_container_width=True,
                        type="primary"
                    )

            with st.expander("Processing Summary", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Segments", task.get("total_segments", 0))
                c2.metric("Needed Speed-up", len(task.get("sped_up", [])))
                c3.metric("Truncated", len(task.get("truncated", [])))

                if task.get("truncated"):
                    st.write("**Truncated:**", ", ".join([f"{n} (cut {m}ms)" for n, m in task["truncated"]]))
                if task.get("sped_up"):
                    st.write("**Sped up:**", ", ".join([f"{n} ({s:.2f}x)" for n, s in task["sped_up"]]))

            if st.button("Process another file"):
                if task_id in st.session_state.tasks:
                    del st.session_state.tasks[task_id]
                remove_from_queue(task_id)
                st.session_state.my_task_id = None
                st.rerun()

        elif task.get("status") == "error":
            st.error(f"Error: {task.get('error')}")
            if st.button("Try again"):
                if task_id in st.session_state.tasks:
                    del st.session_state.tasks[task_id]
                remove_from_queue(task_id)
                st.session_state.my_task_id = None
                st.rerun()

        if st.button("Cancel"):
            if task_id in st.session_state.tasks:
                del st.session_state.tasks[task_id]
            remove_from_queue(task_id)
            st.session_state.my_task_id = None
            st.rerun()

    # INSTRUÇÕES
    with st.expander("How to use this tool (for non-technical users)"):
        st.markdown("""
        1. Select your voice on the first screen.
        2. Upload your .srt file.
        3. Adjust Stability and Similarity if needed (defaults are good).
        4. Click **Generate Dubbed Audio**.
        5. If there is a queue, wait for your turn.
        6. Download the final audio when ready.
        """)

    st.caption("Made with ❤️ for faithful content creators • Powered by ElevenLabs + Streamlit")
