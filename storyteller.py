# requirements(.txt) - 
# streamlit>=1.30.0
# requests>=2.31.0

import streamlit as st
import requests
import os
import textwrap


def call_deepseek(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """
    Calls the DeepSeek API (OpenAI-compatible endpoint).
    NOTE: This uses a placeholder DeepSeek endpoint and model that follow an OpenAI-style schema.
    The endpoint/model name and response parsing may need to be adjusted to match DeepSeek's real API
    (e.g. different model names like 'deepseek-r1' or different response structure).
    """
    api_key = os.getenv("DEEPSEEK_API_KEY") or st.secrets.get("deepseek_api_key")
    if not api_key:
        raise ValueError("DeepSeek API key is missing. Please provide it via the DEEPSEEK_API_KEY "
                         "environment variable or in .streamlit/secrets.toml as deepseek_api_key.")

    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  # placeholder
    DEEPSEEK_MODEL = "deepseek-chat"  # placeholder

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature
    }

    response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
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
    """Builds the detailed user prompt for the DeepSeek model."""
    # Safety truncation
    if len(content) > 8000:
        content = content[:8000] + "\n\n[TRUNCATED FOR DEMO]"

    # Handle empty objectives
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

    st.title("Story Mode – Proof of Concept")
    st.caption("Turn any lesson into an immersive educational story episode using DeepSeek")

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
            placeholder="e.g. Understand how photosynthesis works, Identify the main parts of a plant cell...",
            help="Leave blank and the model will infer objectives automatically"
        )

    with col2:
        st.subheader("Story Settings")
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

        # API key check (graceful error before spinner)
        api_key = os.getenv("DEEPSEEK_API_KEY") or st.secrets.get("deepseek_api_key")
        if not api_key:
            st.error("🚫 DeepSeek API key not found.\n\n"
                     "Set the environment variable **DEEPSEEK_API_KEY** or add "
                     "**deepseek_api_key** to your `.streamlit/secrets.toml` file.")
            st.stop()

        with st.spinner("Generating story episode with DeepSeek..."):
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
                story = call_deepseek(system_prompt, user_prompt, temperature)
                st.subheader("✅ Generated Episode")
                st.markdown(story)
            except Exception as e:
                st.error(f"❌ Error calling DeepSeek API: {str(e)}")


if __name__ == "__main__":
    main()
