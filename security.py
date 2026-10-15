import re
from typing import Optional, Tuple, List, Dict

class InputSanitizer:
    """Detects adversarial prompt injections and cleans delimiter exploits."""

    INJECTION_PATTERNS = [
        # Common English jailbreak and instruction overrides
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(all\s+)?previous",
        r"new\s+instructions\s*:",
        r"system\s*prompt",
        r"---\s*end\s*(of)?\s*prompt",
        r"pretend\s+you\s+are",
        r"act\s+as\s+(if\s+)?you",
        r"bypass\s+(all\s+)?restrictions",
        r"you\s+are\s+now\s+in\s+DAN\s+mode",
        # Common Vietnamese instruction override attempts
        r"bỏ\s+qua\s+(hết\s+)?(tất\s+cả\s+)?(các\s+)?hướng\s+dẫn\s+trước",
        r"quên\s+(hết\s+)?(tất\s+cả\s+)?(các\s+)?lệnh\s+trước",
        r"tiết\s+lộ\s+system\s*prompt",
        r"hãy\s+đóng\s+vai\s+là\s+DAN",
    ]

    def __init__(self):
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.INJECTION_PATTERNS
        ]

    def check(self, text: str) -> Tuple[bool, Optional[str]]:
        """Validate whether the incoming message is free from prompt injections."""
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return False, "Blocked: Potential prompt injection or system override detected."
        return True, None

    def clean(self, text: str) -> str:
        """Strip dangerous structural delimiters that could alter model execution context."""
        cleaned = re.sub(r'[-]{3,}', '', text)
        cleaned = re.sub(r'[=]{3,}', '', cleaned)
        cleaned = cleaned.replace('{{', '{ {').replace('}}', '} }')
        cleaned = re.sub(r'<\/?(?:system|prompt|context)>', '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()


class PIIDetector:
    """Detects and redacts Personally Identifiable Information (PII) before storage or LLM dispatch."""

    PATTERNS: Dict[str, str] = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone_vn": r"\b(?:\+?84|0)(?:3[2-9]|5[689]|7[06-9]|8[1-9]|9[0-9])[0-9]{7}\b",
        "phone_intl": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "citizen_id": r"\b\d{12}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    MASK_MAP: Dict[str, str] = {
        "email": "[EMAIL REDACTED]",
        "phone_vn": "[PHONE REDACTED]",
        "phone_intl": "[PHONE REDACTED]",
        "credit_card": "[CARD REDACTED]",
        "citizen_id": "[CITIZEN_ID REDACTED]",
        "ip_address": "[IP REDACTED]",
    }

    CREDENTIAL_PATTERN = re.compile(
        r'(?i)(\b(?:mật\s*khẩu|password|passwd|pass|mã\s*pin|pin\s*code|api[_\s]?key|secret[_\s]?key|access[_\s]?token)\b[^:\n=]{0,40}?(?:là|is|:|=)\s*)(\S+)'
    )

    def detect(self, text: str) -> Dict[str, List[str]]:
        """Identify any PII entities or sensitive credentials present within the text."""
        findings = {}
        for entity_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                findings[entity_type] = matches

        cred_matches = self.CREDENTIAL_PATTERN.findall(text)
        if cred_matches:
            findings["credentials"] = [m[1] for m in cred_matches]

        return findings

    def mask(self, text: str) -> str:
        """Replace all identified PII entities and passwords with standard redaction placeholders."""
        masked_text = text
        for entity_type, pattern in self.PATTERNS.items():
            masked_text = re.sub(pattern, self.MASK_MAP[entity_type], masked_text)

        masked_text = self.CREDENTIAL_PATTERN.sub(r'\g<1>[PASSWORD REDACTED]', masked_text)
        return masked_text


class OutputValidator:
    """Validates and filters model responses before streaming to client."""

    HARMFUL_PATTERNS = [
        re.compile(r"(?:google|tavily|gemini|api)[_\s]?key\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{15,}", re.IGNORECASE),
        re.compile(r"password\s+is\s+['\"]?\S+['\"]?", re.IGNORECASE),
        re.compile(r"here('s|is) (how|the way) to (hack|exploit|attack)\s+", re.IGNORECASE),
    ]

    def __init__(self):
        self.pii_detector = PIIDetector()

    def validate(self, output_text: str) -> Tuple[str, List[str]]:
        """Sanitize output and verify against information leakage."""
        warnings: List[str] = []

        # Scrub accidental PII leakage in generated text
        detected_pii = self.pii_detector.detect(output_text)
        if detected_pii:
            output_text = self.pii_detector.mask(output_text)
            warnings.append(f"PII masked in output: {list(detected_pii.keys())}")

        # Intercept accidental secret/key leakage
        for pattern in self.HARMFUL_PATTERNS:
            if pattern.search(output_text):
                output_text = "[Response redacted: Content violates system safety guidelines]"
                warnings.append("Harmful or sensitive content intercepted in model output.")
                break

        return output_text, warnings


class SecurityPipeline:
    """Centralized safety pipeline governing agent input/output lifecycle."""

    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.pii_detector = PIIDetector()
        self.output_validator = OutputValidator()

    def check_input(self, text: str) -> Tuple[bool, str, List[str]]:
        """
        Execute pre-flight security checks on incoming user query.
        Returns: (is_allowed, sanitized_text, audit_notes)
        """
        audit_notes: List[str] = []

        # Step 1: Adversarial injection screening
        is_safe, rejection_reason = self.sanitizer.check(text)
        if not is_safe:
            return False, "", [rejection_reason or "Blocked: Prompt injection detected."]

        # Step 2: Context delimiter scrubbing
        cleaned_text = self.sanitizer.clean(text)

        # Step 3: PII redaction before storage and LLM consumption
        detected_pii = self.pii_detector.detect(cleaned_text)
        if detected_pii:
            cleaned_text = self.pii_detector.mask(cleaned_text)
            audit_notes.append(f"Input PII masked: {list(detected_pii.keys())}")

        return True, cleaned_text, audit_notes

    def check_output(self, text: str) -> Tuple[str, List[str]]:
        """Execute post-generation security and leak validation."""
        return self.output_validator.validate(text)
