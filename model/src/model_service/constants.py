"""Dataset and label constants."""

PCAM_DATASET_NAME = "patch_camelyon"

# PCam: 0 = non-metastatic, 1 = metastatic (lymph node metastasis)
LABEL_NAMES = ("non_metastatic", "metastatic")

# Human-readable aliases for API / UI (align with handoff spec wording)
LABEL_ALIASES = {
    "non_metastatic": "no-cancer",
    "metastatic": "cancer",
}
