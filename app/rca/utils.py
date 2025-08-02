from typing import List
def unique_components(meta: List[dict]) -> List[str]:
    seen = set()
    out = []
    for m in meta:
        c = (m.get("component") or "").strip()
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    out.sort()
    return out
