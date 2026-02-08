"""
Shared helpers for singer_attributes.json: alias map so names from singers.txt
(or folder names) can resolve to attribute keys. Without this, mismatches like
'Olivia Rodrigo' vs 'Olivia' give empty tag sets → tag IoU is 0 for those
artists → similarity labels become mostly 0 or 1 instead of continuous.
"""
# Map: name as it appears in data (singers.txt / paths) -> key in singer_attributes.json
SINGER_ATTRIBUTE_ALIASES = {
    "Olivia Rodrigo": "Olivia",
    # Add more as needed, e.g. "Sabrina Carpenter": "Sarbina Carpenter",
}


def resolve_artist_for_attributes(artist_name: str) -> str:
    """Return the key to use for singer_attributes lookup (after alias)."""
    s = (artist_name or "").strip()
    return SINGER_ATTRIBUTE_ALIASES.get(s, s)
