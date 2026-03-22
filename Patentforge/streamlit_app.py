"""
IP-Native Venture Engine — Streamlit web app.
Deploy on Streamlit Community Cloud from the GitHub repo.
"""

import sys
import os

# Make sure ip_venture_engine is importable when run from this directory
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from ip_venture_engine.domains import list_domains, get_domain_by_id
from ip_venture_engine.engine import load_patent_texts, run_engine

# ── Inject API key from Streamlit secrets into the environment ───────────────
# Works for both local (.streamlit/secrets.toml) and Streamlit Cloud (Secrets UI).
try:
    key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if key and not key.startswith("your-"):
        os.environ["ANTHROPIC_API_KEY"] = key
except Exception:
    key = ""

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IP-Native Venture Engine",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Debug: show key status ───────────────────────────────────────────────────
if key and not key.startswith("your-"):
    st.sidebar.success(f"API key loaded: {key[:12]}…")
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "say ok"}],
        )
        st.sidebar.success("API call OK!")
        from ip_venture_engine.engine import process_patent
        test_result = process_patent("A method for cooling data centers using immersion.", {
            "domain_label": "AI infrastructure",
            "patent_start_year": 1980,
            "patent_end_year": 2015,
            "current_year": 2026,
        }, api_key=key)
        if "No API key" in test_result.get("summary", ""):
            err = test_result.get("_error", "no api_key in environ")
            st.sidebar.error(f"Fallback! Error: {err[:150]}")
        else:
            st.sidebar.success(f"process_patent OK: {test_result['summary'][:60]}")
    except Exception as e:
        st.sidebar.error(f"API call failed: {e}")
else:
    st.sidebar.error(f"No valid API key. Value: '{key[:20] if key else 'empty'}'")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Tighten top padding */
  .block-container { padding-top: 1.5rem; }

  /* Domain card button styling */
  div[data-testid="stButton"] > button {
    text-align: left;
    white-space: normal;
    height: auto;
    padding: .45rem .85rem;
  }

  /* Result expand header */
  .result-header {
    background: #1e2530;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: .65rem 1rem;
    margin-bottom: .4rem;
    font-size: .92rem;
  }

  /* Opportunity box */
  .opp-box {
    background: rgba(63,185,80,.1);
    border-left: 3px solid #3fb950;
    border-radius: 0 6px 6px 0;
    padding: .75rem 1rem;
    font-size: .9rem;
    line-height: 1.55;
  }

  /* Step badge */
  .step-badge {
    display: inline-block;
    background: #1f6feb;
    color: white;
    border-radius: 99px;
    font-size: .72rem;
    font-weight: 700;
    padding: .15rem .6rem;
    margin-right: .5rem;
    vertical-align: middle;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ───────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 0          # 0=menu  1=domain/years  2=confirm  3=results
if "context" not in st.session_state:
    st.session_state.context = None
if "results" not in st.session_state:
    st.session_state.results = []
if "selected_domain_id" not in st.session_state:
    st.session_state.selected_domain_id = None

def go(step: int):
    st.session_state.step = step

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💡 IP-Native\nVenture Engine")
    st.markdown("---")

    # Step progress
    labels = ["Menu", "Domain & Years", "Confirm", "Results"]
    for i, label in enumerate(labels):
        icon = "✅" if i < st.session_state.step else ("▶️" if i == st.session_state.step else "○")
        st.markdown(f"{icon} **{label}**" if i == st.session_state.step else f"{icon} {label}")

    st.markdown("---")
    if st.session_state.step > 0:
        if st.button("↩ Back to Menu"):
            st.session_state.step = 0
            st.session_state.context = None
            st.session_state.results = []
            st.rerun()

    st.markdown("---")
    st.caption("Prototype · standard library + Streamlit")

# ── Header ───────────────────────────────────────────────────────────────────
st.title("IP-Native Venture Engine")
st.caption("Identify venture opportunities hidden in patent landscapes.")
st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# SCREEN 0 — Main Menu
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 0:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("What would you like to do?")
        st.markdown(" ")
        if st.button("▶  Start new analysis session", use_container_width=True, type="primary"):
            go(1)
            st.rerun()

    with col2:
        st.subheader("Available domains")
        for d in list_domains():
            st.markdown(f"**{d['id']:>2}.** {d['label']}")

# ════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — Domain & Years
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 1:
    st.markdown('<span class="step-badge">Step 1 of 3</span> **Select Domain & Year Range**',
                unsafe_allow_html=True)
    st.markdown(" ")

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown("#### Choose a domain")

        keyword = st.text_input("Filter by keyword", placeholder="e.g. energy, robotics, pharma")

        domains = list_domains()
        if keyword:
            domains = [d for d in domains if keyword.lower() in d["label"].lower()]
            if not domains:
                st.warning(f'No domains matched "{keyword}".')

        domain_options = {f"{d['id']}. {d['label']}": d["id"] for d in domains}
        selected_label = st.radio(
            "Domains",
            options=list(domain_options.keys()),
            index=None,
            label_visibility="collapsed",
        )

        if selected_label:
            st.session_state.selected_domain_id = domain_options[selected_label]

    with col_right:
        st.markdown("#### Year range")

        year_start = st.number_input("Patent filing start year", min_value=1800, max_value=2100,
                                      value=1980, step=1)
        year_end   = st.number_input("Patent filing end year",   min_value=1800, max_value=2100,
                                      value=2015, step=1)
        year_eval  = st.number_input("Current evaluation year",  min_value=1800, max_value=2200,
                                      value=2026, step=1)

        st.markdown(" ")
        ready = st.session_state.selected_domain_id is not None
        if not ready:
            st.info("Select a domain on the left to continue.")

        if st.button("Next →", type="primary", disabled=not ready, use_container_width=True):
            if year_end < year_start:
                st.error("End year must be ≥ start year.")
            elif year_eval < year_start:
                st.error("Evaluation year must be ≥ start year.")
            else:
                domain = get_domain_by_id(st.session_state.selected_domain_id)
                st.session_state.context = {
                    "domain_id":          domain["id"],
                    "domain_label":       domain["label"],
                    "patent_start_year":  int(year_start),
                    "patent_end_year":    int(year_end),
                    "current_year":       int(year_eval),
                }
                go(2)
                st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — Confirm & Run
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    ctx = st.session_state.context
    st.markdown('<span class="step-badge">Step 2 of 3</span> **Confirm & Run**',
                unsafe_allow_html=True)
    st.markdown(" ")

    # Context summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Domain", f"[{ctx['domain_id']}]")
    c2.metric("Start year",  ctx["patent_start_year"])
    c3.metric("End year",    ctx["patent_end_year"])
    c4.metric("Eval year",   ctx["current_year"])
    st.caption(f"**{ctx['domain_label']}**")

    st.markdown("---")

    # Patent list
    patents = load_patent_texts()

    if not patents:
        st.error("No `.txt` files found in `patents/` directory. Add some patent files and try again.")
    else:
        st.markdown(f"**{len(patents)} patent file(s) found:**")
        for fname, _ in patents:
            st.markdown(f"- 📄 `{fname}`")

        st.markdown(" ")
        col_back, col_run = st.columns([1, 2])
        with col_back:
            if st.button("← Back"):
                go(1)
                st.rerun()
        with col_run:
            if st.button("▶  Run analysis", type="primary", use_container_width=True):
                with st.spinner("Running engine…"):
                    st.session_state.results = run_engine(patents, ctx, api_key=key)
                go(3)
                st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — Results
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    ctx     = st.session_state.context
    results = st.session_state.results

    st.markdown('<span class="step-badge">Step 3 of 3</span> **Results**',
                unsafe_allow_html=True)
    st.markdown(" ")

    if not results:
        st.info("No results to display.")
    else:
        # Context summary row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patents analysed", len(results))
        c2.metric("Start year",  ctx["patent_start_year"])
        c3.metric("End year",    ctx["patent_end_year"])
        c4.metric("Eval year",   ctx["current_year"])
        st.caption(f"Domain: **{ctx['domain_label']}**")
        st.markdown("---")

        for item in results:
            res      = item["result"]
            summary  = res.get("summary", res.get("preview", "")).replace("\n", " ")
            concepts = res.get("concepts", [])
            scores   = res.get("scores", {})
            ms = scores.get("market_size", "—")
            tf = scores.get("technical_feasibility", "—")
            df = scores.get("defensibility", "—")
            first_title = concepts[0].get("title", "—") if concepts else "—"

            with st.expander(f"📄 **{item['filename']}** — {first_title}"):

                # ── Summary ──
                st.markdown("#### Summary")
                st.info(summary)

                # ── Scores ──
                st.markdown("#### Scores")
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Market size",          f"{ms}/10")
                sc2.metric("Technical feasibility", f"{tf}/10")
                sc3.metric("Defensibility",         f"{df}/10")

                # ── Assumptions & changes ──
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("#### Original assumptions")
                    for a in res.get("original_assumptions", []):
                        st.markdown(f"- {a}")

                with col_b:
                    st.markdown(f"#### What changed by {ctx['current_year']}")
                    for c in res.get("changes_by_current_year", []):
                        st.markdown(f"- {c}")

                # ── Concepts ──
                st.markdown("#### Venture concepts")
                for i, concept in enumerate(concepts, 1):
                    with st.container():
                        st.markdown(f"**{i}. {concept.get('title', '—')}**")
                        st.markdown(concept.get("description", ""))

                        ci1, ci2 = st.columns(2)
                        with ci1:
                            st.markdown("**Ideal customer**")
                            st.markdown(concept.get("ideal_customer", "—"))
                            st.markdown("**Why now**")
                            st.markdown(concept.get("why_now", "—"))
                        with ci2:
                            st.markdown("**Moat**")
                            st.markdown(
                                f'<div class="opp-box">{concept.get("moat", "—")}</div>',
                                unsafe_allow_html=True,
                            )
                        if i < len(concepts):
                            st.markdown("---")

        st.markdown("---")
        if st.button("↩ Start new analysis", type="primary"):
            st.session_state.step = 0
            st.session_state.context = None
            st.session_state.results = []
            st.session_state.selected_domain_id = None
            st.rerun()
