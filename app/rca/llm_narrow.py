# app/rca/llm_narrow.py
from typing import List, Dict, Any, Set
import os, json
from openai import OpenAI

client = OpenAI(api_key=(os.getenv("OPENAI_API_KEY") or "").strip())

PROMPT = """You are assisting a maintenance technician. You will see 2–5 candidate fault cases
(component/model + fault description + root cause + corrective action) and a free-text user query.
Propose ONE short yes/no question that best discriminates between these candidates. Also provide
3–6 lowercase keywords to look for in text that indicate a "Yes" answer.

Rules:
- Keep the question 8–16 words, unambiguous, technician-friendly.
- Avoid jargon or multi-part questions.
- The question MUST target a different signal than any previously asked questions.
- NONE of the keywords may match any banned keywords (exact string match).
- Return JSON ONLY with keys: question, keywords (array of strings), rationale.
- Do NOT include any other text.
"""

def _overlaps_banned(kws: List[str], banned: Set[str]) -> bool:
    return any((k or "").strip().lower() in banned for k in kws)

def _similar_question(q: str, banned_qs: Set[str]) -> bool:
    qn = (q or "").strip().lower()
    if not qn:
        return False
    return any(qn == b or qn in b or b in qn for b in banned_qs)

def propose_question(
    query: str,
    candidates: List[Dict[str, Any]],
    banned_keywords: List[str] | None = None,
    banned_questions: List[str] | None = None,
) -> Dict[str, Any]:
    banned_kw_set: Set[str] = { (k or "").strip().lower() for k in (banned_keywords or []) if str(k).strip() }
    banned_q_set: Set[str] = { (q or "").strip().lower() for q in (banned_questions or []) if str(q).strip() }

    ctx = {
        "query": query,
        "candidates": [
            {
                "component": c.get("component"),
                "model": c.get("model") or "",
                "fault_description": c.get("matched_fault_description") or c.get("fault_description"),
                "root_cause": c.get("root_cause"),
                "corrective_action": c.get("corrective_action"),
            } for c in candidates
        ],
        "banned_keywords": sorted(list(banned_kw_set)),
        "banned_questions": sorted(list(banned_q_set)),
    }

    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": json.dumps(ctx, ensure_ascii=False)}
    ]

    for _ in range(3):
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
            max_tokens=250,
        )
        try:
            data = json.loads(res.choices[0].message.content)
        except Exception:
            data = {"question": None, "keywords": [], "rationale": "parse_error"}

        question = (data.get("question") or "").strip()
        keywords = [str(k).strip().lower() for k in (data.get("keywords") or []) if str(k).strip()]
        if question and not _similar_question(question, banned_q_set) and not _overlaps_banned(keywords, banned_kw_set):
            return {"question": question, "keywords": keywords[:6], "rationale": data.get("rationale", "")}

        messages.append({"role": "system", "content": "The previous suggestion overlapped with banned items. Ask about a DIFFERENT symptom and provide NEW, distinct keywords."})

    return {"question": None, "keywords": [], "rationale": "no_non_overlapping_question"}

def apply_answer(answer_yes: bool, keywords: List[str], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Re-score candidates using simple keyword evidence:
      +0.07 per keyword hit on Yes; -0.07 per hit on No (cap +/- 0.15 total)
    Search in merged text: fault + cause + action (lowercased).
    """
    BOOST = 0.07
    CAP = 0.15
    for c in candidates:
        text = " ".join([
            (c.get("matched_fault_description") or c.get("fault_description") or ""),
            c.get("root_cause") or "", c.get("corrective_action") or ""
        ]).lower()
        hits = sum(1 for k in keywords if k and k.lower() in text)
        delta = min(CAP, BOOST * hits)
        if not answer_yes:
            delta = -delta
        c["similarity"] = float(c.get("similarity", 0.0)) + delta
    return sorted(candidates, key=lambda x: x.get("similarity", 0.0), reverse=True)
