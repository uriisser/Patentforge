"""Core processing logic and (future) LLM integration stubs."""

import os


def load_patent_texts(patents_dir: str = "patents") -> list:
    """
    Returns list of (filename, text) for all .txt files in patents_dir.
    Resolves the path relative to this file's directory if not absolute.
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


def process_patent(text: str, context: dict) -> dict:
    """
    Analyse a single patent text against the given context.
    Currently a stub — no external API calls.

    Returns a dict with:
      - preview:            first 150 characters of text
      - context:            the context dict passed in
      - dummy_opportunity:  placeholder venture-opportunity string
    """
    preview = text[:150].strip()
    return {
        "preview": preview,
        "context": context,
        "dummy_opportunity": (
            "Example opportunity idea based on this patent and context. "
            "(Replace with LLM call in engine.py::process_patent.)"
        ),
    }


def run_engine(patents: list, context: dict) -> list:
    """
    Process a list of (filename, text) tuples and return a list of result dicts.

    Each result dict contains:
      - filename
      - result  (dict from process_patent)
    """
    results = []
    for filename, text in patents:
        result = process_patent(text, context)
        results.append({"filename": filename, "result": result})
    return results
