"""BASELINE (first version, GlotLID top-1 + Darija markers), kept only to report its accuracy in the paper.

Language-variety identification for replies: en / fr / msa / ary / ary_latn / other.

GlotLID handles English, French and Arabic-script text but has no label for Darija in Latin
script (Arabizi): on our pilot messages it returned Maltese, Jamaican Creole or Haitian. It
also often labels short Darija as Algerian (arq_Arab). We therefore combine:
  - script detection (Arabic vs Latin letters);
  - GlotLID for en / fr and as one signal for Arabic-script text;
  - Darija lexical markers (Arabic and Latin spellings) and Arabizi digit-letters (3, 7, 9 inside words).
Accuracy on our labelled benchmark messages is reported in the paper (`evaluate()`).
"""
import re

import fasttext
from huggingface_hub import hf_hub_download

MAGHREBI = {"ary_Arab", "arq_Arab", "aeb_Arab"}
DARIJA_AR = re.compile(
    r"(?<!\S)(ديال\S*|واش|بغيت|بغا\S*|فين|شنو|علاش|دابا|كيفاش|ماشي|راه|راني|حيت|عافاك\S*|بزاف|"
    r"كاين\S*|غادي|خاص\S*|ديما|هادشي|داكشي|شحال|مزيان\S*|واخا|بلا ما|ليا|ليك|عندي|ما\S+ش)(?!\S)")
DARIJA_LATN = re.compile(
    r"\b(dyal\w*|dial\w*|wach|bghit|bgha\w*|fin|chno|chnou|3lach|daba|kifach|machi|rah|rani|hit|"
    r"3afak\w*|bzaf|bzzaf|kayn\w*|ghadi|khass\w*|khas\w*|dima|hadchi|dakchi|ch7al|chhal|mzyan\w*|"
    r"wakha|lia|lik|3ndi|ana|nta|nti|hna|chi|walo|bach|m3a|mn|f|w)\b", re.I)
ARABIZI_DIGIT = re.compile(r"[a-z][2379][a-z]|\b[2379][a-z]|[a-z][2379]\b", re.I)

_model = None


def _glotlid(text):
    global _model
    if _model is None:
        _model = fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin"))
    # fasttext's predict() breaks under numpy 2; the C++ binding returns (prob, label) pairs.
    prob, label = _model.f.predict(text.replace("\n", " "), 1, 0.0, "strict")[0]
    return label.replace("__label__", ""), float(prob)


def identify(text):
    """Return (variety, details) for one reply."""
    text = str(text or "").strip()
    arabic = len(re.findall(r"[؀-ۿ]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if arabic + latin == 0:
        return "other", {}
    if arabic >= latin:
        label, prob = _glotlid(text)
        n_markers = len(DARIJA_AR.findall(text))
        density = n_markers / max(1, len(text.split()))
        variety = "ary" if (label in MAGHREBI or density >= 0.08 or n_markers >= 3) else (
            "msa" if label.endswith("_Arab") else "other")
        return variety, {"glotlid": label, "p": round(prob, 3), "markers": n_markers}
    label, prob = _glotlid(text)
    n_markers = len(DARIJA_LATN.findall(text)) + len(ARABIZI_DIGIT.findall(text))
    words = max(1, len(text.split()))
    if n_markers / words >= 0.15 or len(ARABIZI_DIGIT.findall(text)) >= 2:
        variety = "ary_latn"
    elif label == "fra_Latn":
        variety = "fr"
    elif label == "eng_Latn":
        variety = "en"
    else:
        variety = "other"
    return variety, {"glotlid": label, "p": round(prob, 3), "markers": n_markers}


def evaluate(texts, gold):
    """Accuracy and confusion of identify() against known varieties."""
    import pandas as pd
    pred = [identify(t)[0] for t in texts]
    df = pd.DataFrame({"gold": list(gold), "pred": pred})
    return (df.gold == df.pred).mean(), pd.crosstab(df.gold, df.pred)
