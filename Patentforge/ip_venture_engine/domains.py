"""Domain definitions and lookup utilities."""

from typing import Optional

DOMAINS = [
    {"id": 1,  "label": "Humanoid robotics & physical AI"},
    {"id": 2,  "label": "Supply chain, logistics & production floors"},
    {"id": 3,  "label": "AI infrastructure & data centers (power, cooling, siting)"},
    {"id": 4,  "label": "Cross-organizational workflows & contracts"},
    {"id": 5,  "label": "Industrial energy & decarbonization"},
    {"id": 6,  "label": "Mining, materials & heavy industry"},
    {"id": 7,  "label": "Pharma, biotech & metabolic health"},
    {"id": 8,  "label": "National security, cyber & critical infrastructure"},
    {"id": 9,  "label": "Financial systems, risk & narrative dynamics"},
    {"id": 10, "label": "Spatial computing, AR/VR & human-computer interfaces"},
    {"id": 11, "label": "Brain-computer interfaces & assistive tech"},
    {"id": 12, "label": "Government, regulation & public infrastructure"},
]


def list_domains() -> list:
    """Return all domains."""
    return DOMAINS


def find_domains_by_keyword(keyword: str) -> list:
    """Return domains whose label contains keyword (case-insensitive)."""
    kw = keyword.lower()
    return [d for d in DOMAINS if kw in d["label"].lower()]


def get_domain_by_id(domain_id: int) -> Optional[dict]:
    """Return the domain dict with the given id, or None if not found."""
    for d in DOMAINS:
        if d["id"] == domain_id:
            return d
    return None
