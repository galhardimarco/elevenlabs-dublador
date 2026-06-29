import streamlit as st
import os
import subprocess
import numpy as np
import soundfile as sf
import pysrt
import requests
import time
import queue
import threading
import uuid
from datetime import datetime

st.set_page_config(
    page_title="ElevenLabs SRT Voice Generator",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CONFIG =====================
API_KEY = st.secrets["ELEVENLABS_API_KEY"]
HEADERS = {"xi-api-key": API_KEY, "Content-Type": "application/json"}

COUNTRY_VOICES = {
    "Brazil 🇧🇷": {"name": "Brazil Estive New", "id": "YU8EsJtXFMyKMxYtheDk"},
    "Korea 🇰🇷": {"name": "Korean Young Gyu", "id": "h5eZa8VFAq0EQ8E81dfL"},
    "Japanese 🇯🇵": {"name": "Japanese Otani", "id": "3JDquces8E8bkmvbh6Bc"},
    "Filipino 🇵🇭": {"name": "Filipino Pocholo Gonzales", "id": "VCGhAh0uUPumdTXQNdzQ"},
}

# ===================== GLOBAL QUEUE =====================
if "task_queue" not in st.session_state:
    st.session_state.task_queue = queue.Queue()
    st.session_state.tasks = {}
    st.session_state.worker_started = False

def worker():
    while True:
        try:
            job = st.session_state.task_queue.get(timeout=5)
            task_id = job["task_id"]
            st.session_state.tasks[task_id]["status"] = "processing"

            try:
                result = process_audio_job(job)
                st.session_state.tasks[task_id]["status"] = "done"
                st.session_state.tasks[task_id]["result_path"] = result["output_path"]
                st.session_state.tasks[task_id]["filename"] = result["filename"]
            except Exception as e:
                st.session_state.tasks[task_id]["status"] = "error"
                st.session_state.tasks[task_id]["error"] = str(e)

            st.session_state.task_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Worker error: {e}")
            time.sleep(1)

def start_worker_once():
    if not st.session_state.worker_started:
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        st.session_state.worker_started = True

start_worker_once()

# ===================== PROCESSING LOGIC =====================
def generate_tts(text, output_path, voice_id, stability, similarity, model_id):
    data = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {"stability": stability, "similarity_boost": similarity}
    }
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    response = requests.post(url, headers=HEADERS, json=data, timeout=90)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    else:
        raise Exception(f"ElevenLabs Error {response.status_code}")

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

def process_audio_job(job):
    task_id = job["task_id"]
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
    sped_up_segments = []
    truncated_segments = []

    for idx, sub in enumerate(valid_subs):
        text = sub.text.strip().replace("\n", " ")
        if not text:
            continue

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

        do_truncate = False
        speed_used = 1.0
        truncated_amount_ms = 0

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
                truncated_amount_ms = new_dur - available
                audio_data = audio_data[:max_samples]
                do_truncate = True
                truncated_segments.append((idx + 1, truncated_amount_ms))
            elif speed_used > 1.25:
                sped_up_segments.append((idx + 1, speed_used))

        start_sample = int(start_ms / 1000.0 * sr)
        end_sample = start_sample + len(audio_data)

        if end_sample > len(final_audio):
            new_len = end_sample + 2000
            new_audio = np.zeros((new_len, 2), dtype=np.float32)
            new_audio[:len(final_audio)] = final_audio
            final_audio = new_audio

        final_audio[start_sample:end_sample] = audio_data

        if os.path.exists(temp_audio):
            os.remove(temp_audio)

    output_path = f"final_{task_id}.mp3"
    sf.write(output_path, final_audio, sr)
    os.remove(temp_srt)

    return {
        "output_path": output_path,
        "filename": job["filename"].replace(".srt", "_dubbed.mp3"),
        "sped_up_segments": sped_up_segments,
        "truncated_segments": truncated_segments,
        "total_segments": total
    }

@st.cache_data(ttl=300, show_spinner=False)
def get_subscription_info():
    try:
        resp = requests.get("https://api.elevenlabs.io/v1/user/subscription", headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            used = data.get("character_count", 0)
            limit = data.get("character_limit", 0)
            tier = data.get("tier", "Unknown")
            remaining = max(0, limit - used) if limit > 0 else 0
            pct = round((used / limit * 100), 1) if limit > 0 else 0
            return {"tier": tier, "used": used, "limit": limit, "remaining": remaining, "pct": pct, "error": None}
        else:
            return {"error": f"API Error {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# ===================== SEGURANÇA =====================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if not st.session_state.password_correct:
        st.title("🔐 Restricted Access")
        password = st.text_input("Password", type="password")
        if st.button("Login", type="primary", use_container_width=True):
            if password == st.secrets.get("APP_PASSWORD", "default_password"):
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

# ===================== TELA 1: SELEÇÃO DE VOZ =====================
if st.session_state.selected_country is None:
    st.title("🌍 ElevenLabs SRT Voice Generator")
    st.markdown("### Select your voice / language")

    country = st.selectbox("Choose voice", options=list(COUNTRY_VOICES.keys()), index=0)
    st.info("**Recommended for Brazilian Portuguese:** Brazil voice")

    if st.button("Continue →", type="primary", use_container_width=True):
        st.session_state.selected_country = country
        st.session_state.selected_voice_name = COUNTRY_VOICES[country]["name"]
        st.session_state.selected_voice_id = COUNTRY_VOICES[country]["id"]
        st.rerun()

else:
    # ===================== TELA 2: UPLOAD + CONFIG =====================
    st.title("🎙️ ElevenLabs SRT Voice Generator")
    st.caption(f"Voice: **{st.session_state.selected_voice_name}**")

    if st.button("← Change Voice"):
        st.session_state.selected_country = None
        st.rerun()

    # Sidebar
    with st.sidebar:
        st.header("📊 Account")
        sub = get_subscription_info()
        if sub.get("error"):
            st.error(sub["error"])
        else:
            st.metric("Plan", sub["tier"])
            c1, c2 = st.columns(2)
            c1.metric("Used", f"{sub['used']:,}")
            c2.metric("Remaining", f"{sub['remaining']:,}")
            st.progress(sub["pct"] / 100, text=f"{sub['pct']}% used")

        st.divider()
        st.header("⚙️ Settings")
        force_fit = st.checkbox("Force segments to fit timing (truncate if needed)", value=False,
                                help="Only enable for strict timing. May cut the end of sentences. Recommended: OFF for teaching content.")
        model_id = st.selectbox("Model", ["eleven_v3", "eleven_turbo_v2_5"], index=0)

    # Upload + Preview
    uploaded_file = st.file_uploader("Upload .srt file", type=["srt"])

    if uploaded_file:
        try:
            temp_path = f"preview_{int(time.time())}.srt"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            subs = pysrt.open(temp_path, encoding='utf-8')
            valid = [s for s in subs if s.text.strip()]
            total_chars = sum(len(s.text.strip()) for s in valid)
            total_segs = len(valid)

            st.info(f"**{total_segs} segments** • **{total_chars:,} characters** • Estimated cost (v3): **${total_chars/1000 * 0.10:.2f}**")

            os.remove(temp_path)
        except Exception as e:
            st.warning(f"Could not preview file: {e}")

    st.subheader("🎛️ Voice Quality")
    col1, col2 = st.columns(2)
    with col1:
        stability = st.slider("Stability", 0.0, 1.0, 0.60, 0.05)
    with col2:
        similarity = st.slider("Similarity Boost", 0.0, 1.0, 0.85, 0.05)

    # ===================== GERAÇÃO =====================
    if st.session_state.my_task_id is None:
        if uploaded_file and st.button("🚀 Generate Audio", type="primary", use_container_width=True):
            task_id = str(uuid.uuid4())[:8]

            job = {
                "task_id": task_id,
                "file_bytes": uploaded_file.getbuffer().tobytes(),
                "filename": uploaded_file.name,
                "voice_id": st.session_state.selected_voice_id,
                "stability": stability,
                "similarity": similarity,
                "model_id": model_id,
                "force_fit": force_fit,
            }

            st.session_state.tasks[task_id] = {
                "status": "queued",
                "created_at": datetime.now()
            }

            st.session_state.task_queue.put(job)
            st.session_state.my_task_id = task_id
            st.rerun()

    else:
        # Usuário já tem task
        task_id = st.session_state.my_task_id
        task = st.session_state.tasks.get(task_id, {})

        if task.get("status") == "queued":
            qsize = st.session_state.task_queue.qsize()
            st.info(f"⏳ Your file is in queue. **{qsize}** file(s) ahead of you.")
            if st.button("🔄 Refresh"):
                st.rerun()

        elif task.get("status") == "processing":
            st.warning("🎤 Processing your file right now...")
            if st.button("🔄 Refresh"):
                st.rerun()

        elif task.get("status") == "done":
            st.success("✅ Audio generated successfully!")

            result_path = task.get("result_path")
            if result_path and os.path.exists(result_path):
                with open(result_path, "rb") as f:
                    st.download_button(
                        "📥 Download Final Dubbed Audio",
                        data=f,
                        file_name=task.get("filename", "dubbed.mp3"),
                        mime="audio/mpeg",
                        use_container_width=True,
                        type="primary"
                    )

            # Resumo
            if task.get("total_segments"):
                with st.container(border=True):
                    st.subheader("📋 Processing Summary")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Segments", task.get("total_segments", 0))
                    c2.metric("Needed Speed-up", len(task.get("sped_up_segments", [])))
                    c3.metric("Truncated", len(task.get("truncated_segments", [])))

                    if task.get("truncated_segments"):
                        trunc_text = ", ".join([f"**{n}** (cut {m}ms)" for n, m in task["truncated_segments"]])
                        st.markdown(f"- Truncated: {trunc_text}")
                    if task.get("sped_up_segments"):
                        sped_text = ", ".join([f"**{n}** ({s:.2f}x)" for n, s in task["sped_up_segments"]])
                        st.markdown(f"- Sped up: {sped_text}")

            if st.button("🔄 Process another file"):
                if task_id in st.session_state.tasks:
                    del st.session_state.tasks[task_id]
                st.session_state.my_task_id = None
                st.rerun()

        elif task.get("status") == "error":
            st.error(f"Error: {task.get('error')}")
            if st.button("Try again"):
                if task_id in st.session_state.tasks:
                    del st.session_state.tasks[task_id]
                st.session_state.my_task_id = None
                st.rerun()

        if st.button("← Cancel"):
            if task_id in st.session_state.tasks:
                del st.session_state.tasks[task_id]
            st.session_state.my_task_id = None
            st.rerun()

st.divider()
st.caption("Multi-user version with queue • Made for content creators")
