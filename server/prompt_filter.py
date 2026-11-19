import re
import unicodedata


_ROLE_MARKERS = re.compile(
    r"<\|(?:im_start|im_end|start_header_id|end_header_id|system|assistant|user)\|>"
    r"|\[/?inst\]|</?system\s*>|<</?sys>>"
)
_OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass|discard)\s+"
    r"(?:(?:all|the|any|your|my|prior|previous|earlier|above|original|system|developer|safety)\s+){0,8}"
    r"(?:instructions?|prompts?|rules?|polic(?:y|ies)|guidelines?)\b"
)
_DISCLOSURE = re.compile(
    r"\b(?:show|print|reveal|repeat|output|disclose|expose|display|give|tell|share|return|provide)\s+"
    r"(?:(?:me|us|your|the|full|entire|exact|original|all|verbatim|text|of)\s+){0,8}"
    r"(?:system|developer|internal|hidden)\s+(?:prompts?|instructions?|messages?)\b"
)


def looks_like_prompt_injection(content: str) -> bool:
    normalized = unicodedata.normalize("NFKC", content).casefold()
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Cf")
    return any(pattern.search(normalized) for pattern in (_ROLE_MARKERS, _OVERRIDE, _DISCLOSURE))
