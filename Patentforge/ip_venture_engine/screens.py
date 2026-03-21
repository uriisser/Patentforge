"""All terminal screens / UI flows for IP-Native Venture Engine."""

import os
import sys
from typing import Optional

from .domains import list_domains, find_domains_by_keyword, get_domain_by_id
from .engine import load_patent_texts, run_engine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HEADER = "IP-Native Venture Engine (Prototype)"
DIVIDER = "-" * 60


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def pause(message: str = "Press Enter to continue...") -> None:
    input(f"\n{message}")


def _print_header(subtitle: str = "") -> None:
    clear_screen()
    print(DIVIDER)
    print(f"  {HEADER}")
    if subtitle:
        print(f"  {subtitle}")
    print(DIVIDER)


def _read_int(prompt: str, min_val: int = None, max_val: int = None) -> int:
    """Keep asking until a valid integer (within optional bounds) is entered."""
    while True:
        raw = input(prompt).strip()
        if not raw.isdigit() and not (raw.startswith("-") and raw[1:].isdigit()):
            print("  Please enter a valid integer.")
            continue
        val = int(raw)
        if min_val is not None and val < min_val:
            print(f"  Value must be >= {min_val}.")
            continue
        if max_val is not None and val > max_val:
            print(f"  Value must be <= {max_val}.")
            continue
        return val


# ---------------------------------------------------------------------------
# Screen 1 -- Main Menu
# ---------------------------------------------------------------------------

def main_menu() -> int:
    """
    Show landing screen and return user choice:
      1 = start new analysis session
      2 = manage domains
      3 = exit
    """
    while True:
        _print_header()
        print()
        print("  Welcome to the IP-Native Venture Engine.")
        print("  Identify venture opportunities hidden in patent landscapes.\n")
        print("  1) Start new analysis session")
        print("  2) Manage domains")
        print("  3) Exit")
        print()
        choice = input("  Enter choice [1-3]: ").strip()
        if choice in ("1", "2", "3"):
            return int(choice)
        print("  Invalid choice. Please enter 1, 2, or 3.")
        pause()


# ---------------------------------------------------------------------------
# Screen 2 -- Domain & Years Selection
# ---------------------------------------------------------------------------

def _show_domain_list() -> None:
    print()
    print("  Available domains:")
    print()
    for d in list_domains():
        print(f"    {d['id']:>2})  {d['label']}")
    print()


def screen_select_domain_and_years() -> Optional[dict]:
    """
    Let the user pick a domain and enter year range / evaluation year.
    Returns context dict or None if the user cancels.
    """
    chosen_domain = None

    while chosen_domain is None:
        _print_header("Step 1 of 3 -- Select Domain & Years")
        _show_domain_list()

        print("  1) Choose domain by number")
        print("  2) Search domain by keyword")
        print("  3) Cancel")
        print()
        nav = input("  Enter choice [1-3]: ").strip()

        if nav == "3":
            return None

        elif nav == "1":
            domain_id = _read_int(
                "  Enter domain ID: ",
                min_val=1,
                max_val=len(list_domains()),
            )
            domain = get_domain_by_id(domain_id)
            if domain is None:
                print(f"  No domain with ID {domain_id}.")
                pause()
                continue
            chosen_domain = domain

        elif nav == "2":
            keyword = input("  Enter search keyword: ").strip()
            if not keyword:
                print("  Keyword cannot be empty.")
                pause()
                continue
            matches = find_domains_by_keyword(keyword)
            if not matches:
                print(f'  No domains matched "{keyword}".')
                pause()
                continue
            print()
            print(f'  Matches for "{keyword}":')
            print()
            for d in matches:
                print(f"    {d['id']:>2})  {d['label']}")
            print()
            domain_id = _read_int(
                "  Enter domain ID from the list above: ",
                min_val=1,
                max_val=len(list_domains()),
            )
            domain = get_domain_by_id(domain_id)
            if domain is None or domain not in matches:
                print(f"  ID {domain_id} is not in the search results.")
                pause()
                continue
            chosen_domain = domain

        else:
            print("  Invalid choice.")
            pause()
            continue

    # Year inputs
    print()
    print(f'  Selected domain: [{chosen_domain["id"]}] {chosen_domain["label"]}')
    print()
    patent_start_year = _read_int(
        "  Enter patent filing start year (e.g. 1980): ", min_val=1800, max_val=2100
    )
    patent_end_year = _read_int(
        "  Enter patent filing end year   (e.g. 2015): ",
        min_val=patent_start_year,
        max_val=2100,
    )
    current_year = _read_int(
        "  Enter current evaluation year  (e.g. 2026): ",
        min_val=patent_start_year,
        max_val=2200,
    )

    return {
        "domain_id": chosen_domain["id"],
        "domain_label": chosen_domain["label"],
        "patent_start_year": patent_start_year,
        "patent_end_year": patent_end_year,
        "current_year": current_year,
    }


# ---------------------------------------------------------------------------
# Screen 3 -- Confirm & Run
# ---------------------------------------------------------------------------

def screen_confirm_and_run(context: dict) -> list:
    """
    Load patent texts, confirm with the user, then run the engine.
    Returns list of result dicts (may be empty).
    """
    _print_header("Step 2 of 3 -- Confirm & Run")
    patents = load_patent_texts()

    if not patents:
        print()
        print("  No .txt files found in the patents/ directory.")
        print("  Add patent text files and try again.")
        pause()
        return []

    print()
    print(f"  Found {len(patents)} patent file(s):\n")
    for i, (fname, _) in enumerate(patents[:10], 1):
        print(f"    {i:>2})  {fname}")
    if len(patents) > 10:
        print(f"       ... and {len(patents) - 10} more.")
    print()
    print("  Context summary:")
    print(f"    Domain : [{context['domain_id']}] {context['domain_label']}")
    print(f"    Years  : {context['patent_start_year']} - {context['patent_end_year']}")
    print(f"    Eval   : {context['current_year']}")
    print()

    answer = input("  Run analysis on these patents? (y/n): ").strip().lower()
    if answer != "y":
        print("  Analysis cancelled.")
        pause()
        return []

    print()
    print("  Running engine", end="", flush=True)
    results = run_engine(patents, context)
    print(" done.")
    print(f"  Processed {len(results)} patent(s).")
    pause()
    return results


# ---------------------------------------------------------------------------
# Screen 4 -- Results Viewer
# ---------------------------------------------------------------------------

def _truncate(text: str, width: int = 55) -> str:
    return text if len(text) <= width else text[:width - 1] + "..."


def _score_bar(value: int, total: int = 10, width: int = 10) -> str:
    """Return a simple ASCII bar, e.g. [######    ] 6/10"""
    filled = round(value / total * width)
    bar = "#" * filled + "." * (width - filled)
    return f"[{bar}] {value}/{total}"


def screen_show_results(results: list, context: dict) -> None:
    """
    Show a numbered list of results and allow drilling into individual items.
    """
    if not results:
        _print_header("Results")
        print()
        print("  No results to display.")
        pause()
        return

    while True:
        _print_header("Step 3 of 3 -- Results")
        print()
        print(f"  {len(results)} result(s) for domain: {context['domain_label']}\n")

        for i, item in enumerate(results, 1):
            res     = item["result"]
            summary = res.get("summary", res.get("preview", "")).replace("\n", " ")
            concept = (res.get("concepts") or [{}])[0]
            title   = concept.get("title", "—")
            scores  = res.get("scores", {})
            ms = scores.get("market_size", "?")
            tf = scores.get("technical_feasibility", "?")
            df = scores.get("defensibility", "?")
            print(f"  {i:>2})  {item['filename']}")
            print(f"       Summary : {_truncate(summary, 65)}")
            print(f"       Concept : {_truncate(title, 65)}")
            print(f"       Scores  : Market {ms}/10  |  Feasibility {tf}/10  |  Defensibility {df}/10")
            print()

        print("  Enter result number to view details, or 'b' to go back:")
        nav = input("  > ").strip().lower()

        if nav == "b":
            return

        if nav.isdigit():
            idx = int(nav) - 1
            if 0 <= idx < len(results):
                _show_result_detail(results[idx], context)
                continue

        print("  Invalid input.")
        pause()


def _show_result_detail(item: dict, context: dict) -> None:
    """Show full detail for a single result."""
    res = item["result"]
    while True:
        _print_header("Result Detail")
        print()
        print(f"  File   : {item['filename']}")
        print(f"  Domain : [{context['domain_id']}] {context['domain_label']}")
        print(f"  Years  : {context['patent_start_year']} - {context['patent_end_year']}  |  Eval: {context['current_year']}")
        print()

        # Summary
        print(f"  SUMMARY")
        print(f"  {res.get('summary', '—')}")
        print()

        # Original assumptions
        assumptions = res.get("original_assumptions", [])
        if assumptions:
            print("  ORIGINAL ASSUMPTIONS")
            for a in assumptions:
                print(f"    - {a}")
            print()

        # What changed
        changes = res.get("changes_by_current_year", [])
        if changes:
            print(f"  WHAT CHANGED BY {context['current_year']}")
            for c in changes:
                print(f"    - {c}")
            print()

        # Concepts
        concepts = res.get("concepts", [])
        for idx, concept in enumerate(concepts, 1):
            print(f"  CONCEPT {idx}: {concept.get('title', '—')}")
            print(f"    Description    : {concept.get('description', '—')}")
            print(f"    Ideal customer : {concept.get('ideal_customer', '—')}")
            print(f"    Why now        : {concept.get('why_now', '—')}")
            print(f"    Moat           : {concept.get('moat', '—')}")
            print()

        # Scores
        scores = res.get("scores", {})
        if scores:
            print("  SCORES")
            print(f"    Market size          : {_score_bar(scores.get('market_size', 1))}")
            print(f"    Technical feasibility: {_score_bar(scores.get('technical_feasibility', 1))}")
            print(f"    Defensibility        : {_score_bar(scores.get('defensibility', 1))}")
            print()

        print("  'b' to return to results list.")
        nav = input("  > ").strip().lower()
        if nav == "b":
            return
