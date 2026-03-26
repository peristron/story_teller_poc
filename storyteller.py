import streamlit as st
import requests
import os
import textwrap


def call_llm(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7
) -> str:
    """
    Calls the chosen LLM API (all are OpenAI-compatible).
    API keys are loaded from Streamlit Secrets (set in the Community Cloud dashboard).
    
    Supported providers: DeepSeek, OpenAI, Grok
    NOTE: Model names and endpoints are current as of March 2026.
          You can adjust model names in the dictionary below if newer versions are released.
    """
    # Load key from secrets (Streamlit Cloud preferred method)
    api_key = st.secrets.get(f"{provider.lower()}_api_key")
    if not api_key:
        raise ValueError(f"🚫 {provider} API key not found in Streamlit Secrets. "
                         f"Please add '{provider.lower()}_api_key' in the dashboard.")

    # Provider configuration
    config = {
        "DeepSeek": {
            "url": "https://api.deepseek.com/v1/chat/completions",
            "model": "deepseek-chat"
        },
        "OpenAI": {
            "url": "https://api.openai.com/v1/chat/completions",
            "model": "gpt-4o-mini"
        },
        "Grok": {
            "url": "https://api.x.ai/v1/chat/completions",
            "model": "grok-beta"
        }
    }

    if provider not in config:
        raise ValueError(f"Unsupported provider: {provider}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

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


def build_story_prompt(
    content: str,
    objectives: str,
    genre: str,
    tone: str,
    age_band: str,
    scenes: int = 3,
    questions_per_scene: int = 1,
) -> str:
    """Builds the detailed user prompt for the LLM (unchanged from original)."""
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
    st.set_page_config(
        page_title="Story Mode – Proof of Concept",
        layout="wide"
    )

    # ===================== PASSWORD PROTECTION =====================
    # The password is stored securely in Streamlit Secrets (set in the Community Cloud dashboard)
    # Format in Secrets (TOML):
    # password = "your-chosen-password-here"
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        st.title("🔒 Story Mode – Proof of Concept")
        st.markdown("**Protected access** — enter the password set in Streamlit Secrets to continue.")

        password_input = st.text_input(
            "Enter app password",
            type="password",
            key="pw_input"
        )

        if st.button("🔓 Unlock App", type="primary", use_container_width=True):
            stored_password = st.secrets.get("password")
            if stored_password and password_input == stored_password:
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("❌ Incorrect password. Please try again (or ask the owner to check Secrets).")
        st.stop()  # Prevent rest of app from rendering

    # ===================== MAIN APP UI =====================
    st.title("Story Mode – Proof of Concept")
    st.caption("Turn any lesson into an immersive educational story episode")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Lesson Input")
        lesson_content = st.text_area(
            "Lesson Content",
            height=300,
            placeholder="Paste the full lesson text or topic here...",
            help="The educational concepts that will be embedded in the story"
        )
        learning_objectives = st.text_area(
            "Learning Objectives (optional)",
            height=120,
            placeholder="e.g. Understand how photosynthesis works...",
            help="Leave blank and the model will infer objectives automatically"
        )

    with col2:
        st.subheader("Story Settings")
        
        provider = st.selectbox(
            "AI Provider (API key must be in Secrets)",
            ["DeepSeek", "OpenAI", "Grok"],
            help="DeepSeek = deepseek_api_key | OpenAI = openai_api_key | Grok = grok_api_key"
        )
        
        genre = st.selectbox(
            "Genre",
            ["Fantasy academy", "Space mission", "Mystery / investigation", "Slice of life", "Superhero"]
        )
        tone = st.selectbox(
            "Tone",
            ["Serious", "Light / playful", "Humorous but not silly"]
        )
        age_band = st.selectbox(
            "Age band",
            ["Upper elementary (10–12)", "Middle school (12–14)", "High school (14–18)", "Adult learners"]
        )

        num_scenes = st.slider("Number of scenes", 2, 5, 3)
        questions_per_scene = st.slider("Questions per scene", 1, 3, 1)
        temperature = st.slider("Creativity (temperature)", 0.1, 1.0, 0.7, step=0.1)

    if st.button("Generate Story Episode", type="primary", use_container_width=True):
        if not lesson_content.strip():
            st.warning("⚠️ Lesson Content cannot be empty.")
            st.stop()

        # Quick check that the chosen provider has a key in secrets
        if not st.secrets.get(f"{provider.lower()}_api_key"):
            st.error(f"🚫 {provider} API key not found in Streamlit Secrets.\n\n"
                     f"Add **{provider.lower()}_api_key** in the app settings on Streamlit Cloud.")
            st.stop()

        with st.spinner(f"Generating story episode with {provider}..."):
            system_prompt = "You are a careful, pedagogy-aware narrative designer for educational content."
            user_prompt = build_story_prompt(
                lesson_content,
                learning_objectives,
                genre,
                tone,
                age_band,
                num_scenes,
                questions_per_scene
            )

            try:
                story = call_llm(provider, system_prompt, user_prompt, temperature)
                st.subheader(f"✅ Generated Episode (using {provider})")
                st.markdown(story)
            except Exception as e:
                st.error(f"❌ Error calling {provider} API: {str(e)}")


if __name__ == "__main__":
    main()
