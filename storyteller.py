import streamlit as st
import requests
import os
import textwrap
import io
import asyncio
import tempfile
from pathlib import Path
from openai import OpenAI
import PyPDF2
import docx
import edge_tts


def call_llm(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7
) -> str:
    secret_mapping = {
        "DeepSeek": "DEEPSEEK_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
        "Grok": "XAI_API_KEY"
    }
    secret_name = secret_mapping.get(provider)
    if not secret_name:
        raise ValueError(f"Unsupported provider: {provider}")

    api_key = st.secrets.get(secret_name)
    if not api_key:
        raise ValueError(f"🚫 {secret_name} not found in Streamlit Secrets.")

    config = {
        "DeepSeek": {"url": "https://api.deepseek.com/v1/chat/completions", "model": "deepseek-chat"},
        "OpenAI": {"url": "https://api.openai.com/v1/chat/completions", "model": "gpt-4o-mini"},
        "Grok": {"url": "https://api.x.ai/v1/chat/completions", "model": "grok-beta"}
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": config[provider]["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature
    }

    response = requests.post(config[provider]["url"], json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def extract_text_from_files(uploaded_files) -> str:
    parts = []
    for file in uploaded_files:
        try:
            name = file.name.lower()
            raw = file.getvalue()
            if name.endswith(".pdf"):
                reader = PyPDF2.PdfReader(io.BytesIO(raw))
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        parts.append(text.strip())
            elif name.endswith(".docx"):
                doc = docx.Document(io.BytesIO(raw))
                for p in doc.paragraphs:
                    if p.text.strip():
                        parts.append(p.text.strip())
            elif name.endswith(".txt"):
                parts.append(raw.decode("utf-8", errors="replace").strip())
            else:
                parts.append(f"[Unsupported file type: {file.name}]")
        except Exception as e:
            parts.append(f"[Error reading {file.name}: {e}]")
    return "\n\n---\n\n".join(parts)


def _sync_edge_tts(text: str, voice: str = "en-US-GuyNeural", speed: float = 1.0) -> bytes:
    try:
        rate_str = f"{int((speed - 1) * 100):+d}%"
        communicate = edge_tts.Communicate(text, voice, rate=rate_str)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            loop.run_until_complete(communicate.save(tmp_path))
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            Path(tmp_path).unlink(missing_ok=True)
            return audio_bytes
        finally:
            loop.close()
    except Exception as e:
        st.error(f"Edge TTS failed: {e}")
        return b""


def build_story_prompt(
    content: str,
    objectives: str,
    genre: str,
    tone: str,
    age_band: str,
    scenes: int = 3,
    questions_per_scene: int = 1,
) -> str:
    if len(content) > 8000:
        content = content[:8000] + "\n\n[TRUNCATED FOR DEMO]"

    objectives_text = objectives.strip() or "Infer the key learning objectives from the content."

    prompt = textwrap.dedent(f"""\
    You are an educational narrative designer.

    Create ONE short story episode that embeds the educational concepts from the provided lesson content.

    Target audience: {age_band}
    Genre: {genre}
    Tone: {tone}
    Number of scenes: exactly {scenes}
    Questions per scene: exactly {questions_per_scene}

    Learning objectives:
    {objectives_text}

    Source lesson content (embed concepts naturally — do NOT copy long sections verbatim):
    ```content
    {content}
    ```

    Output format — follow this structure exactly (plain text with clear headings):

    EPISODE TITLE: <short, engaging title>

    SCENE 1: <scene title>
    Narrative:
    <2–4 paragraphs of story text>

    Question(s):
    1. <question text phrased naturally in-world as a puzzle or decision>
       - Type: MCQ or OPEN
       - Options (if MCQ): a) ..., b) ..., c) ...
       - Correct answer: ...
       - Explanation: ...

    (repeat for each scene)

    Constraints:
    • Exactly {scenes} scenes.
    • Each scene introduces or reinforces 1–2 concepts from the source content.
    • Questions must feel like natural in-world puzzles/decisions.
    • Language must be age-appropriate for {age_band}.
    • Keep the total story concise and engaging.
    • Never break character or mention that this is educational content.
    """)
    return prompt.strip()


def main():
    st.set_page_config(page_title="Story Mode – Proof of Concept", layout="wide")

    # ===================== PASSWORD PROTECTION =====================
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        st.title("🔒 Story Mode – Proof of Concept")
        st.markdown("**Protected access** — enter the password set in Streamlit Secrets to continue.")

        stored_pw = st.secrets.get("APP_PASSWORD")
        if not stored_pw:
            st.warning("⚠️ No **APP_PASSWORD** secret found in Streamlit Secrets!")

        password_input = st.text_input("Enter app password", type="password", key="pw_input")

        if st.button("🔓 Unlock App", type="primary", use_container_width=True):
            stored_password = st.secrets.get("APP_PASSWORD")
            if stored_password and password_input.strip() == str(stored_password).strip():
                st.session_state.password_correct = True
                st.success("✅ Password correct! Loading app...")
                st.rerun()
            else:
                st.error("❌ Incorrect password.")
        st.stop()

    # ===================== SESSION STATE =====================
    if "lesson_content" not in st.session_state:
        st.session_state.lesson_content = ""
    if "generated_story" not in st.session_state:
        st.session_state.generated_story = ""
    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None

    # ===================== MAIN UI =====================
    st.title("Story Mode – Proof of Concept")
    st.caption("Turn any lesson into an immersive educational story episode")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Lesson Input")
        uploaded_files = st.file_uploader(
            "📤 Upload lesson files (PDF, DOCX, TXT) — multiple supported",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"]
        )

        if uploaded_files and st.button("📤 Extract & Append Files", type="secondary", use_container_width=True):
            with st.spinner("Extracting text from uploaded files..."):
                extracted = extract_text_from_files(uploaded_files)
                if extracted.strip():
                    st.session_state.lesson_content = (st.session_state.lesson_content + "\n\n" + extracted).strip()
                    st.success(f"✅ Extracted text from {len(uploaded_files)} file(s) and appended!")
                    st.rerun()
                else:
                    st.warning("No text could be extracted from the files.")

        lesson_content = st.text_area(
            "Lesson Content (paste or edit here — optional if files were uploaded)",
            value=st.session_state.lesson_content,
            height=300
        )
        if lesson_content != st.session_state.lesson_content:
            st.session_state.lesson_content = lesson_content

        learning_objectives = st.text_area(
            "Learning Objectives (optional)",
            height=120,
            placeholder="e.g. Understand how photosynthesis works..."
        )

    with col2:
        st.subheader("Story Settings")
        provider = st.selectbox("AI Provider", ["DeepSeek", "OpenAI", "Grok"])

        cost_map = {"DeepSeek": "~ $0.005", "OpenAI": "~ $0.02", "Grok": "~ $0.03"}
        st.info(f"**💰 Rough cost per generation**\n"
                f"• DeepSeek: {cost_map['DeepSeek']}\n"
                f"• OpenAI: {cost_map['OpenAI']}\n"
                f"• Grok: {cost_map['Grok']}")

        genre = st.selectbox("Genre", ["Fantasy academy", "Space mission", "Mystery / investigation", "Slice of life", "Superhero"])
        tone = st.selectbox("Tone", ["Serious", "Light / playful", "Humorous but not silly"])
        age_band = st.selectbox("Age band", ["Upper elementary (10–12)", "Middle school (12–14)", "High school (14–18)", "Adult learners"])

        num_scenes = st.slider("Number of scenes", 2, 5, 3)
        questions_per_scene = st.slider("Questions per scene", 1, 3, 1)
        temperature = st.slider("Creativity (temperature)", 0.1, 1.0, 0.7, step=0.1)

    if st.button("Generate Story Episode", type="primary", use_container_width=True):
        if not st.session_state.lesson_content.strip():
            st.warning("⚠️ Please either paste lesson content OR upload and extract files first.")
            st.stop()

        secret_mapping = {"DeepSeek": "DEEPSEEK_API_KEY", "OpenAI": "OPENAI_API_KEY", "Grok": "XAI_API_KEY"}
        if not st.secrets.get(secret_mapping[provider]):
            st.error(f"🚫 {secret_mapping[provider]} not found in Streamlit Secrets.")
            st.stop()

        with st.spinner(f"Generating story episode with {provider}..."):
            system_prompt = "You are a careful, pedagogy-aware narrative designer for educational content."
            user_prompt = build_story_prompt(
                st.session_state.lesson_content,
                learning_objectives,
                genre,
                tone,
                age_band,
                num_scenes,
                questions_per_scene
            )
            try:
                story = call_llm(provider, system_prompt, user_prompt, temperature)
                st.session_state.generated_story = story
                st.session_state.audio_bytes = None  # clear old audio when new story is made
                st.success("✅ Story generated!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # ===================== READ-ALOUD SECTION (now persistent) =====================
    if st.session_state.generated_story:
        st.divider()
        st.subheader("🔊 Read the Episode Aloud")

        tts_provider = st.radio(
            "TTS Provider",
            ["OpenAI TTS (paid, higher quality)", "Edge TTS (Free)"],
            horizontal=True
        )

        if st.button("🎙️ Generate Full Audio", type="primary", use_container_width=True):
            story_text = st.session_state.generated_story
            clean_text = story_text.replace("**", "").replace("Question(s):", "\n\n[Question]\n")

            with st.spinner("Generating audio..."):
                if tts_provider.startswith("OpenAI"):
                    openai_key = st.secrets.get("OPENAI_API_KEY")
                    if not openai_key:
                        st.error("OpenAI API key required for OpenAI TTS.")
                    else:
                        try:
                            client = OpenAI(api_key=openai_key)
                            response = client.audio.speech.create(
                                model="tts-1", voice="alloy", input=clean_text[:4000]
                            )
                            st.session_state.audio_bytes = response.content
                        except Exception as e:
                            st.error(f"OpenAI TTS error: {e}")
                else:
                    audio_bytes = _sync_edge_tts(clean_text[:4000])
                    if audio_bytes:
                        st.session_state.audio_bytes = audio_bytes
                    else:
                        st.error("Edge TTS failed to generate audio.")

                st.rerun()  # refresh to show the player

        # Persistent audio player + download (outside the button)
        if st.session_state.audio_bytes:
            st.audio(st.session_state.audio_bytes, format="audio/mp3")
            st.download_button(
                "⬇️ Download MP3",
                st.session_state.audio_bytes,
                file_name="story_episode.mp3",
                mime="audio/mpeg",
                use_container_width=True
            )
            st.caption("✅ Audio is now saved in session — you can play or download without regenerating.")

        st.subheader("📖 Generated Episode")
        st.markdown(st.session_state.generated_story)

    else:
        st.info("Generate a story episode first ↑")

    if st.session_state.generated_story and st.button("🗑️ Clear Story & Start Over"):
        st.session_state.generated_story = ""
        st.session_state.audio_bytes = None
        st.rerun()


if __name__ == "__main__":
    main()
