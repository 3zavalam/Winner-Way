import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import re

load_dotenv()  # Para leer tu API key desde un .env

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_stroke_json(keypoints_folder: str) -> dict:
    stroke_data = {}
    for frame in ["preparation", "impact", "follow_through"]:
        json_path = os.path.join(keypoints_folder, f"{frame}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                stroke_data[frame] = json.load(f)
        else:
            stroke_data[frame] = None
    return stroke_data

def summarize_keypoints(data: dict) -> str:
    summary = []
    for phase, points in data.items():
        if not points:
            summary.append(f"{phase}: missing")
            continue
        coords = [(round(p['x'], 3), round(p['y'], 3)) for p in points]
        summary.append(f"{phase}: {coords}")
    return "\n".join(summary)


def analyze_stroke_with_ai(stroke_json: dict, stroke_type: str) -> list:
    prompt = f"""You are a professional tennis coach helping a player improve their {stroke_type} technique.
Below is the 2D joint coordinate data (keypoints) for three stroke phases: preparation, impact, and follow-through.

{summarize_keypoints(stroke_json)}

Your task:
- First, list 1 to 2 good elements in the stroke using the ✔️ emoji.
- Then, list 2 to 3 areas for improvement using the ⚠️ emoji.
- Each point must start with the corresponding emoji and a space.
- Keep all points in a single-line bullet list, with each line starting with "- ".

Example:
- ✔️ Preparation is well-timed and allows smooth transitions.
- ✔️ Good shoulder rotation adds power.
- ⚠️ Limited hip turn reduces energy transfer.
- ⚠️ Wrist position at contact is unstable.

Return only the bullet points, no explanation or extra formatting.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a tennis biomechanics expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=500
        )
        raw = response.choices[0].message.content.strip()

        points = [
            line[2:].strip()
            for line in raw.split('\n')
            if line.strip().startswith('- ')
        ]
        return points
    except Exception as e:
        return [f"❌ Error calling OpenAI: {e}"]


def generate_drills_with_ai(issues: list, stroke_type: str) -> list:
    prompt = (
        f"You are a professional tennis coach.\n"
        f"The player has the following issues with their {stroke_type}:\n"
    )
    issues_list = "\n".join(f"- {issue}" for issue in issues)
    prompt += issues_list + """\n
For each issue:
- Suggest a drill with a short title.
- Provide a concise description of the drill (1 line).
- Include 2–3 bullet point steps to fix the issue.

Return ONLY a VALID JSON array. Format strictly like this:

[
  {
    "title": "Brief issue summary",
    "drill": "Drill Name",
    "steps": [
      "Step one",
      "Step two"
    ]
  }
]

Do NOT include any commentary, markdown, explanation, or intro. Only raw JSON, nothing else.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in tennis drills."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=800
        )

        content = response.choices[0].message.content.strip() if response.choices else None

        # Guardar para depuración
        with open("drill_debug.txt", "w") as f:
            f.write(content or "NO CONTENT")

        print("🧠 RAW RESPONSE FROM OPENAI:\n", content)

        if not content:
            raise ValueError("Empty response from OpenAI")

        # Limpiar bloque de código si viene con ```json ... ```
        if content.startswith("```json"):
            content = re.sub(r"^```json\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)

        if not content.strip().startswith("["):
            raise ValueError(f"OpenAI response invalid or not JSON array:\n{content}")

        return json.loads(content)

    except json.JSONDecodeError as jde:
        return [{
            "title": "Error",
            "drill": "AI failed",
            "steps": [f"Invalid JSON from OpenAI: {jde}"]
        }]
    except Exception as e:
        return [{
            "title": "Error",
            "drill": "AI failed",
            "steps": [f"Exception: {e}"]
        }]