"""
Core processing logic and LLM integration.

Set ANTHROPIC_API_KEY in your environment to enable real analysis.
Without a key the module falls back to a clearly-labelled dummy response.
"""

import json
import os

# ── Prompt template ───────────────────────────────────────────────────────────
# Placeholders: {text}, {domain_label}, {patent_start_year},
#               {patent_end_year}, {current_year}

PROMPT_TEMPLATE = """\
You are an expert venture-capital analyst and deep-tech strategist.

Analyse the patent below and identify startup opportunities that become \
feasible by {current_year}, given how the world has changed since the patent \
was originally filed.

## Patent text
{text}

## Context
- Domain: {domain_label}
- Patent filing window: {patent_start_year}–{patent_end_year}
- Current evaluation year: {current_year}

## Instructions
Return ONLY a single JSON object — no markdown fences, no explanation — that \
strictly matches this schema (scores are integers 1–10):

{{
  "summary": "Short technical summary of the invention (2–3 sentences).",
  "original_assumptions": [
    "Assumption the inventors had to make that is no longer true",
    "..."
  ],
  "changes_by_current_year": [
    "Concrete change by {current_year} that affects feasibility or market",
    "..."
  ],
  "concepts": [
    {{
      "title": "Punchy startup name / concept title",
      "description": "2–4 sentences describing the product or service.",
      "ideal_customer": "Who would pay for this and why.",
      "why_now": "What changed that makes this the right moment.",
      "moat": "IP, data, infrastructure, or regulatory advantages."
    }}
  ],
  "scores": {{
    "market_size": 1,
    "technical_feasibility": 1,
    "defensibility": 1
  }}
}}
"""


# ── Dummy fallback ────────────────────────────────────────────────────────────

def _dummy_result(text: str, context: dict) -> dict:
    return {
        "summary": (
            "No API key present — this is a placeholder summary. "
            "Set ANTHROPIC_API_KEY to get real analysis."
        ),
        "original_assumptions": [
            "Placeholder assumption A",
            "Placeholder assumption B",
        ],
        "changes_by_current_year": [
            "Placeholder change X",
            "Placeholder change Y",
        ],
        "concepts": [
            {
                "title": "Example Venture Concept (dummy)",
                "description": "Placeholder product description.",
                "ideal_customer": "Placeholder customer segment.",
                "why_now": "Placeholder timing rationale.",
                "moat": "Placeholder competitive advantage.",
            }
        ],
        "scores": {
            "market_size": 1,
            "technical_feasibility": 1,
            "defensibility": 1,
        },
        "preview": text[:150].strip(),
        "context": context,
    }


# ── Main processing function ──────────────────────────────────────────────────

def process_patent(text: str, context: dict, api_key: str = None) -> dict:
    """
    Analyse a single patent against the given context.

    Returns the standard result dict (see PROMPT_TEMPLATE for schema).
    Falls back to a dummy dict if no API key is set or if the LLM call fails.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        print("Warning: No ANTHROPIC_API_KEY found — using dummy process_patent.")
        return _dummy_result(text, context)

    prompt = PROMPT_TEMPLATE.format(
        text=text,
        domain_label=context["domain_label"],
        patent_start_year=context["patent_start_year"],
        patent_end_year=context["patent_end_year"],
        current_year=context["current_year"],
    )

    try:
        from anthropic import Anthropic  # imported here so the module loads without anthropic installed

        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text
        # Strip markdown fences if the model added them
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)

    except json.JSONDecodeError as exc:
        print(f"Warning: JSON parse error from LLM ({exc}) — falling back to dummy.")
        result = _dummy_result(text, context)

    except Exception as exc:  # network error, auth error, quota, etc.
        print(f"Warning: LLM call failed ({exc}) — falling back to dummy.")
        result = _dummy_result(text, context)
        result["_error"] = str(exc)

    # Always attach preview + context regardless of which path we took
    result.setdefault("preview", text[:150].strip())
    result.setdefault("context", context)
    return result


# ── File loader ───────────────────────────────────────────────────────────────

def load_patent_texts(patents_dir: str = "patents") -> list:
    """
    Returns list of (filename, text) for all .txt files in patents_dir.
    Resolves path relative to this file's directory if not absolute.
    """
    if not os.path.isabs(patents_dir):
        base = os.path.dirname(os.path.abspath(__file__))
        patents_dir = os.path.join(base, patents_dir)

    results = []
    if not os.path.isdir(patents_dir):
        return results

    for fname in sorted(os.listdir(patents_dir)):
        if fname.lower().endswith(".txt"):
            fpath = os.path.join(patents_dir, fname)
            with open(fpath, "r", encoding="utf-8") as fh:
                text = fh.read()
            results.append((fname, text))

    return results


# ── Engine runner ─────────────────────────────────────────────────────────────

def run_engine(patents: list, context: dict, api_key: str = None) -> list:
    """
    Process a list of (filename, text) tuples.
    Returns list of {"filename": str, "result": dict}.
    """
    results = []
    for filename, text in patents:
        result = process_patent(text, context, api_key=api_key)
        results.append({"filename": filename, "result": result})
    return results
