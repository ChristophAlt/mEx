import os


LANGUAGE = os.getenv("MACSS_MEDICAL_IE_LANGUAGE", "de")

NER_MODEL_PATH = os.getenv("MACSS_MEDICAL_IE_NER_MODEL_PATH",
    raise ValueError("'MACSS_MEDICAL_IE_NER_MODEL_PATH' must be specified."))

RE_MODEL_PATH = os.getenv("MACSS_MEDICAL_IE_RE_MODEL_PATH",
    raise ValueError("'MACSS_MEDICAL_IE_RE_MODEL_PATH' must be specified."))

NEGATION_TRIGGER_PATH = os.getenv("MACSS_MEDICAL_IE_NEGATION_TRIGGER_PATH",
    raise ValueError("'MACSS_MEDICAL_IE_NEGATION_TRIGGER_PATH' must be specified."))
