import io
import json
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document
from lxml import etree

# Optional AI libraries
from openai import OpenAI
import anthropic


# =========================
# Utility: Text Extraction
# =========================

def extract_text_from_file(uploaded_file) -> str:
    """
    Extract text from an uploaded file.

    Supported extensions:
      - .txt, .md: treated as plain text
      - .html, .htm: HTML parsed to text with BeautifulSoup
      - .pdf: PDF parsed with pdfminer.six
      - .docx: Word doc parsed with python-docx
    """
    filename = uploaded_file.name.lower()
    suffix = Path(filename).suffix

    if suffix in [".txt", ".md"]:
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if suffix in [".html", ".htm"]:
        html = uploaded_file.read().decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n")

    if suffix == ".pdf":
        data = uploaded_file.read()
        with io.BytesIO(data) as bio:
            text = pdf_extract_text(bio)
        return text

    if suffix == ".docx":
        data = uploaded_file.read()
        with io.BytesIO(data) as bio:
            doc = Document(bio)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs)

    # Fallback: treat as text
    return uploaded_file.read().decode("utf-8", errors="ignore")


# =========================
# Utility: Lesson Splitting
# =========================

def split_text_into_lessons(full_text: str, num_lessons: int) -> List[str]:
    """
    Split full_text into num_lessons chunks by paragraph, approximately evenly.

    This is a naive splitter, but good enough as a starting point.
    You can replace this with a more sophisticated approach (or LLM planning).
    """
    paragraphs = [p.strip() for p in full_text.split("\n") if p.strip()]
    if not paragraphs:
        return [full_text]

    if num_lessons <= 1 or len(paragraphs) <= num_lessons:
        chunks: List[str] = []
        per_chunk = max(1, len(paragraphs) // max(1, num_lessons))
        for i in range(0, len(paragraphs), per_chunk):
            chunks.append("\n\n".join(paragraphs[i: i + per_chunk]))
        return chunks

    total = len(paragraphs)
    base_size = total // num_lessons
    remainder = total % num_lessons

    lesson_texts: List[str] = []
    start = 0
    for i in range(num_lessons):
        size = base_size + (1 if i < remainder else 0)
        end = start + size
        lesson_paras = paragraphs[start:end]
        if lesson_paras:
            lesson_texts.append("\n\n".join(lesson_paras))
        start = end

    if not lesson_texts:
        return [full_text]

    return lesson_texts


# =========================
# Utility: AI Caller
# =========================

def detect_available_providers() -> List[str]:
    """
    Look at st.secrets and return a list of available providers.
    Expected secrets structure:
      [deepseek]
      api_key = "..."
      base_url = "https://api.deepseek.com"  # optional

      [anthropic]
      api_key = "..."

      [openai]
      api_key = "..."
    """
    providers = ["None (use raw text only)"]

    # DeepSeek (OpenAI-compatible)
    if "deepseek" in st.secrets and st.secrets["deepseek"].get("api_key"):
        providers.append("DeepSeek")

    # Anthropic
    if "anthropic" in st.secrets and st.secrets["anthropic"].get("api_key"):
        providers.append("Anthropic")

    # OpenAI
    if "openai" in st.secrets and st.secrets["openai"].get("api_key"):
        providers.append("OpenAI")

    return providers


def call_llm_for_lesson_html(
    provider: str,
    model: str,
    course_title: str,
    lesson_title: str,
    body_text: str,
    include_quiz: bool,
) -> Optional[dict]:
    """
    Call the selected LLM provider to transform body_text into structured lesson content.

    Returns a dict like:
    {
      "html_body": "<p>...</p>",
      "quiz_html": "<h2>...</h2>..."
    }

    or None if provider is "None" or something goes wrong.
    """
    if provider.startswith("None"):
        return None

    # Prompt template: keep it focused and deterministic.
    system_prompt = (
        "You are an instructional designer. Given raw source text, you will create:\n"
        "1) Well-structured HTML body content for a lesson (no <html> or <body> wrapper).\n"
        "2) An optional HTML quiz/reflection section, simple and self-contained.\n\n"
        "Return ONLY valid JSON with keys 'html_body' and 'quiz_html'."
    )

    user_prompt = f"""
Course title: {course_title}
Lesson title: {lesson_title}

Source text:
\"\"\"
{body_text}
\"\"\"

Requirements:
- Use <h2>, <h3>, <p>, <ul>, <li> as appropriate.
- Keep the language clear and concise.
- For quiz_html, include either:
  - A short reflection prompt with a <textarea>, or
  - 2-3 simple questions in <ul><li> format.
- Do not include <html>, <head>, or <body> tags; only inner content fragments.
"""

    try:
        if provider == "DeepSeek":
            cfg = st.secrets["deepseek"]
            api_key = cfg.get("api_key")
            base_url = cfg.get("base_url", "https://api.deepseek.com")
            client = OpenAI(api_key=api_key, base_url=base_url)

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            content = resp.choices[0].message.content

        elif provider == "OpenAI":
            cfg = st.secrets["openai"]
            api_key = cfg.get("api_key")
            base_url = cfg.get("base_url", None)
            if base_url:
                client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                client = OpenAI(api_key=api_key)

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            content = resp.choices[0].message.content

        elif provider == "Anthropic":
            cfg = st.secrets["anthropic"]
            api_key = cfg.get("api_key")
            client = anthropic.Anthropic(api_key=api_key)

            resp = client.messages.create(
                model=model,
                max_tokens=2048,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                system=system_prompt,
            )
            content_blocks = resp.content
            content = ""
            for block in content_blocks:
                if block.type == "text":
                    content += block.text

        else:
            return None

        # Expect JSON in content
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if "\n" in content:
                first_line, rest = content.split("\n", 1)
                if first_line.strip().lower() in ("json", "jsonc"):
                    content = rest

        data = json.loads(content)
        if not isinstance(data, dict):
            return None

        html_body = data.get("html_body", "")
        quiz_html = data.get("quiz_html", "") if include_quiz else ""

        return {
            "html_body": html_body,
            "quiz_html": quiz_html,
        }

    except Exception as e:
        st.warning(f"AI generation failed for {lesson_title}: {e}")
        return None


# =========================
# Utility: Lesson HTML
# =========================

def generate_lesson_html(
    course_title: str,
    lesson_title: str,
    body_text: str,
    include_quiz: bool,
    provider: str,
    model: str,
) -> str:
    """
    Generate minimal HTML for a single SCO / lesson.

    If an AI provider/model is configured and available, we call it to produce
    structured HTML fragments. Otherwise we fall back to a naive transform.
    """
    import html

    ai_result = call_llm_for_lesson_html(
        provider=provider,
        model=model,
        course_title=course_title,
        lesson_title=lesson_title,
        body_text=body_text,
        include_quiz=include_quiz,
    )

    if ai_result is not None:
        body_html = ai_result.get("html_body", "")
        quiz_html = ai_result.get("quiz_html", "") if include_quiz else ""
    else:
        paragraphs = [p.strip() for p in body_text.split("\n") if p.strip()]
        body_parts = [f"<p>{html.escape(p)}</p>" for p in paragraphs]
        body_html = "".join(body_parts)

        quiz_html = ""
        if include_quiz:
            quiz_html = """
            <hr />
            <h2>Quick Reflection</h2>
            <p>In your own words, summarize the most important idea from this lesson.</p>
            <textarea rows="4" style="width: 100%;"></textarea>
            """

    escaped_course = html.escape(course_title)
    escaped_lesson = html.escape(lesson_title)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>{escaped_course} - {escaped_lesson}</title>
  <script src="../scorm_api_wrapper.js"></script>
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 1.5rem;
      line-height: 1.5;
    }}
    h1, h2, h3 {{
      font-weight: 600;
    }}
    .nav-buttons {{
      margin-top: 2rem;
      display: flex;
      justify-content: flex-end;
    }}
    button {{
      padding: 0.5rem 1rem;
      font-size: 1rem;
    }}
  </style>
</head>
<body onload="scormInit()" onunload="scormTerminate()">
  <h1>{escaped_course}</h1>
  <h2>{escaped_lesson}</h2>

  {body_html}

  {quiz_html}

  <div class="nav-buttons">
    <button type="button" onclick="markCompleteAndExit()">Finish Lesson</button>
  </div>
</body>
</html>
"""
    return html_doc


# =========================
# Utility: SCORM JS Wrapper
# =========================

def get_scorm_api_wrapper_js() -> str:
    """
    Return a minimal SCORM 1.2 API wrapper JS as a string.
    For production, you may want a more complete implementation.
    """
    return """// Minimal SCORM 1.2 API wrapper.
// NOTE: This is intentionally simple and may need to be enhanced
// for production scenarios.

var apiHandle = null;
var initialized = false;

function findAPI(win) {
  var attempts = 0;
  while ((win.API == null) && (win.parent != null) && (win.parent != win) && (attempts < 7)) {
    attempts++;
    win = win.parent;
  }
  return win.API != null ? win.API : null;
}

function getAPI() {
  if (apiHandle != null) {
    return apiHandle;
  }

  var win = window;
  apiHandle = findAPI(win);

  if ((apiHandle == null) && (win.opener != null) && typeof(win.opener) != "undefined") {
    apiHandle = findAPI(win.opener);
  }

  return apiHandle;
}

function scormInit() {
  var api = getAPI();
  if (api == null) {
    console.log("SCORM API not found. Running in non-SCORM context.");
    return;
  }

  var result = api.LMSInitialize("");
  if (result != "true") {
    console.log("LMSInitialize failed.");
  } else {
    initialized = true;
    console.log("SCORM session initialized.");
  }
}

function scormTerminate() {
  var api = getAPI();
  if (api == null || !initialized) {
    return;
  }
  api.LMSFinish("");
  initialized = false;
}

function setValue(element, value) {
  var api = getAPI();
  if (api == null || !initialized) {
    return;
  }
  api.LMSSetValue(element, value);
}

function commit() {
  var api = getAPI();
  if (api == null || !initialized) {
    return;
  }
  api.LMSCommit("");
}

function markCompleteAndExit() {
  try {
    setValue("cmi.core.lesson_status", "completed");
    setValue("cmi.core.score.raw", "100");
    setValue("cmi.core.score.max", "100");
    commit();
  } catch (e) {
    console.log("Error marking completion: " + e);
  }
  scormTerminate();
  try {
    window.close();
  } catch (e) {
    console.log("Unable to close window: " + e);
  }
}
"""


# =========================
# Utility: imsmanifest.xml
# =========================

def generate_imsmanifest_scorm12(
    course_title: str,
    resources: List[Tuple[str, str, str]],
) -> str:
    """
    Generate a minimal SCORM 1.2 imsmanifest.xml as a string.

    Parameters
    ----------
    course_title : str
        Title of the course.
    resources : list of (sco_folder, sco_index_path, lesson_title)
    """
    NS_IMSCP = "http://www.imsproject.org/xsd/imscp_rootv1p1p2"
    NS_ADLCP = "http://www.adlnet.org/xsd/adlcp_rootv1p2"
    NS_XSI = "http://www.w3.org/2001/XMLSchema-instance"

    NSMAP = {
        None: NS_IMSCP,
        "adlcp": NS_ADLCP,
        "xsi": NS_XSI,
    }

    manifest = etree.Element(
        "manifest",
        nsmap=NSMAP,
        identifier="MANIFEST-1",
        version="1.0",
    )

    # <metadata>
    metadata = etree.SubElement(manifest, "metadata")
    schema = etree.SubElement(metadata, "schema")
    schema.text = "ADL SCORM"
    schemaversion = etree.SubElement(metadata, "schemaversion")
    schemaversion.text = "1.2"

    # <organizations>
    organizations = etree.SubElement(manifest, "organizations", default="ORG-1")
    org = etree.SubElement(organizations, "organization", identifier="ORG-1")

    title_el = etree.SubElement(org, "title")
    title_el.text = course_title

    # <resources>
    resources_el = etree.SubElement(manifest, "resources")

    for idx, (sco_folder, sco_index_path, lesson_title) in enumerate(resources, start=1):
        item_id = f"ITEM-{idx}"
        res_id = f"RES-{idx}"

        # Organization item
        item_el = etree.SubElement(
            org,
            "item",
            identifier=item_id,
            identifierref=res_id,
        )
        item_title_el = etree.SubElement(item_el, "title")
        item_title_el.text = lesson_title

        # Resource
        res_el = etree.SubElement(
            resources_el,
            "resource",
            identifier=res_id,
            type="webcontent",
            href=sco_index_path,
        )
        res_el.set("{%s}scormType" % NS_ADLCP, "sco")

        # File entry
        etree.SubElement(res_el, "file", href=sco_index_path)

    xml_bytes = etree.tostring(
        manifest,
        pretty_print=True,
        xml_declaration=True,
        encoding="UTF-8",
    )
    return xml_bytes.decode("utf-8")


# =========================
# Utility: SCORM ZIP builder
# =========================

def build_scorm_package(
    course_title: str,
    lesson_html_list: List[Tuple[str, str]],
) -> io.BytesIO:
    """
    Build a minimal SCORM 1.2-style package and return as a BytesIO.

    Parameters
    ----------
    course_title : str
        Title of the course.
    lesson_html_list : list of (lesson_title, lesson_html)
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Add SCORM runtime wrapper JS at root
        scorm_js = get_scorm_api_wrapper_js()
        zf.writestr("scorm_api_wrapper.js", scorm_js)

        # 2. Add lessons as scoX/index.html
        resource_entries: List[Tuple[str, str, str]] = []
        for idx, (lesson_title, lesson_html) in enumerate(lesson_html_list, start=1):
            sco_folder = f"sco{idx}"
            sco_index_path = f"{sco_folder}/index.html"
            zf.writestr(sco_index_path, lesson_html)
            resource_entries.append((sco_folder, sco_index_path, lesson_title))

        # 3. imsmanifest.xml
        manifest_xml = generate_imsmanifest_scorm12(
            course_title=course_title,
            resources=resource_entries,
        )
        zf.writestr("imsmanifest.xml", manifest_xml)

    zip_buffer.seek(0)
    return zip_buffer


# =========================
# Streamlit App
# =========================

APP_TITLE = "AI-Assisted SCORM 1.2 Package Generator"


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.write(
        "Upload content (text, HTML, PDF, DOCX, Markdown) and generate a minimal "
        "SCORM 1.2 package as a downloadable ZIP. Optionally, use AI providers to "
        "shape the lesson content."
    )

    with st.sidebar:
        st.header("Course Configuration")
        course_title = st.text_input("Course title", value="Generated Course")
        num_lessons = st.number_input(
            "Number of lessons (SCOs)",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
        )
        include_quiz = st.checkbox("Include quiz/reflection section", value=True)

        st.markdown("---")
        st.header("AI Settings")

        providers = detect_available_providers()
        provider_choice = st.selectbox("Provider", providers)

        model_name_default = ""
        if provider_choice == "DeepSeek":
            model_name_default = "deepseek-chat"
        elif provider_choice == "Anthropic":
            model_name_default = "claude-3-opus-20240229"
        elif provider_choice == "OpenAI":
            model_name_default = "gpt-4.1-mini"

        model_name = st.text_input(
            "Model name",
            value=model_name_default,
            help="Model name as expected by the chosen provider. Leave blank to skip AI.",
        )

        st.caption(
            "Configure API keys in Streamlit secrets (e.g., [deepseek], [anthropic], [openai]). "
            "If no provider or model is set, the app will just use the raw text."
        )

        st.markdown("---")
        st.header("Tools")
        st.write(
            "SCORM Inspector:\n\n"
            "[Open SCORM Validator / Inspector](https://os-scorm-inspector.streamlit.app/)"
        )

    st.header("Input Content")

    col_upload, col_paste = st.columns(2)

    uploaded_files = []
    with col_upload:
        st.subheader("Upload files")
        uploaded_files = st.file_uploader(
            "Upload one or more content files",
            type=["txt", "md", "html", "htm", "pdf", "docx"],
            accept_multiple_files=True,
        )

    pasted_text = ""
    with col_paste:
        st.subheader("Or paste raw text")
        pasted_text = st.text_area(
            "Paste content here",
            height=250,
            placeholder="Paste large text content here if you prefer not to upload files.",
        )

    # Button: generate SCORM
    if st.button("Generate SCORM Package", type="primary"):
        if not uploaded_files and not pasted_text.strip():
            st.error("Please upload at least one file or paste some text.")
            st.stop()

        # Normalize provider choice if model is empty
        if not model_name.strip():
            provider_choice_effective = "None (use raw text only)"
        else:
            provider_choice_effective = provider_choice

        with st.spinner("Processing content and generating SCORM package..."):
            # Collect and combine raw text
            all_text_chunks: List[str] = []

            for file in uploaded_files:
                try:
                    text = extract_text_from_file(file)
                    if text.strip():
                        all_text_chunks.append(text)
                except Exception as e:
                    st.warning(f"Could not extract text from {file.name}: {e}")

            if pasted_text.strip():
                all_text_chunks.append(pasted_text.strip())

            if not all_text_chunks:
                st.error("No usable text content could be extracted.")
                st.stop()

            combined_text = "\n\n".join(all_text_chunks)

            # Split into lessons
            lesson_texts = split_text_into_lessons(combined_text, int(num_lessons))

            # Generate HTML per lesson
            lesson_html_list: List[Tuple[str, str]] = []
            for idx, lesson_text in enumerate(lesson_texts, start=1):
                lesson_title = f"Lesson {idx}"
                html_content = generate_lesson_html(
                    course_title=course_title,
                    lesson_title=lesson_title,
                    body_text=lesson_text,
                    include_quiz=include_quiz,
                    provider=provider_choice_effective,
                    model=model_name.strip(),
                )
                lesson_html_list.append((lesson_title, html_content))

            # Build SCORM package ZIP
            zip_buffer = build_scorm_package(
                course_title=course_title,
                lesson_html_list=lesson_html_list,
            )

            # Store in session_state for smooth re-downloads
            st.session_state["scorm_zip"] = zip_buffer.getvalue()

        st.success("SCORM package generated successfully!")

    # Download button rendered whenever a package exists in session_state
    if "scorm_zip" in st.session_state:
        st.download_button(
            label="Download last generated SCORM Package (ZIP)",
            data=st.session_state["scorm_zip"],
            file_name="generated_scorm_package.zip",
            mime="application/zip",
        )

    st.info(
        "You can now upload the generated ZIP as a SCORM 1.2 package into your LMS "
        "(e.g., Brightspace). AI shaping of content is optional and controlled "
        "from the sidebar."
    )


if __name__ == "__main__":
    main()
