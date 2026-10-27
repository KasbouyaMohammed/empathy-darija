"""Language-variety identification for replies: en / fr / msa / ary / ary_latn / other.

GlotLID handles English, French and Arabic-script text but has no label for Darija in Latin
script (Arabizi): on our pilot messages it returned Maltese, Jamaican Creole or Haitian. It
also often labels short Darija as Algerian (arq_Arab). We therefore combine:
  - script detection (Arabic vs Latin letters);
  - GlotLID for en / fr and as one signal for Arabic-script text;
  - Darija lexical markers (Arabic and Latin spellings) and Arabizi digit-letters (3, 7, 9 inside words).
GlotLID's top-1 label is unreliable on short text (Arabic script labelled Angika/Devanagari, short
English labelled Jamaican Creole), so we read its top-k distribution restricted to the closed set of
candidate languages, and weigh Darija markers against Modern Standard Arabic markers.
Rules were designed on the dev split of DarijaBench-derived gold data; accuracy on the held-out test
split is reported in the paper (scripts/04a_langid_validation.py, `evaluate()`).
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
DARIJA_AR_EXTRA = re.compile(
    r"(?<!\S)(اللي|لي|هاد|هادي|هادا|هادو|شي|والو|بلاصة|شوية|كاع|ڭاع|بحال|معا|حتى|ف|غير|دبا|واحد|ديك|"
    r"هاديك|تا|منين|فوقاش|تاني|زعما)(?!\S)|ڭ|ݣ")
MSA_AR = re.compile(
    r"(?<!\S)(هل|يمكن\S*|أريد|نريد|ماذا|لماذا|ليس|ليست|لست|سوف|الذي|التي|الذين|هذا|هذه|ذلك|تلك|لكن|"
    r"لدي\S*|لديك\S*|عندما|حيث|أيضا|أيضًا|جدا|جدًا|إلى|إلي|كم|متى|أين|أنا|نحن|أنت|لقد|قد|كانت|يكون|"
    r"تكون|ثم|بعض|كل)(?!\S)")
DARIJA_LATN_EXTRA = re.compile(
    r"\b(sir|tmma|temma|hna|hnaya|walou|walo|bghina|kan\w*|kay\w*|ka\w+|ma\w+ch|mat\w+|ghir|gha|"
    r"ghan\w*|wa7ed|wahed|chwiya|chkoun|chkun|mnin|imta|ach|achno|ach7al|sme7\w*|sm7\w*|safi|zwin\w*|"
    r"labas|bslama|ntoma|7na|houma|homa|li|lli|had|hadi|hada|hadou|dak|dik|l|b|m3ak|3lik|3lia|dyali|"
    r"dyalek)\b", re.I)
ARABIZI_DIGIT = re.compile(r"[a-z][235789][a-z]|\b[235789][a-z]|[a-z][235789]\b", re.I)

_model = None


def _glotlid(text, k=50):
    """GlotLID top-k as {label: prob}."""
    global _model
    if _model is None:
        _model = fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin"))
    # fasttext's predict() breaks under numpy 2; the C++ binding returns (prob, label) pairs.
    pairs = _model.f.predict(text.replace("\n", " "), k, 0.0, "strict")
    return {label.replace("__label__", ""): float(prob) for prob, label in pairs}


def _arabic_mass(probs):
    """Probability mass on Maghrebi Arabic vs. other Arabic varieties (MSA and Mashriqi)."""
    maghrebi = sum(p for l, p in probs.items() if l in MAGHREBI)
    other = sum(p for l, p in probs.items()
                if l.endswith("_Arab") and l[:2] in ("ar", "ac", "ap", "aj", "ay", "af", "ab", "ae")
                and l not in MAGHREBI)
    return maghrebi, other


def identify(text):
    """Return (variety, details) for one text."""
    text = str(text or "").strip()
    arabic = len(re.findall(r"[؀-ۿ]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if arabic + latin == 0:
        return "other", {}
    probs = _glotlid(text)
    top = max(probs, key=probs.get)
    if arabic >= latin:
        n_darija = len(DARIJA_AR.findall(text)) + len(DARIJA_AR_EXTRA.findall(text))
        n_msa = len(MSA_AR.findall(text))
        maghrebi, other = _arabic_mass(probs)
        if n_darija != n_msa:
            variety = "ary" if n_darija > n_msa else "msa"
        else:
            variety = "ary" if maghrebi > other else "msa"
        return variety, {"glotlid": top, "p": round(probs[top], 3), "markers": n_darija,
                         "msa_markers": n_msa, "p_maghrebi": round(maghrebi, 3)}
    n_digit = len(ARABIZI_DIGIT.findall(text))
    n_markers = len(DARIJA_LATN.findall(text)) + len(DARIJA_LATN_EXTRA.findall(text)) + n_digit
    p_en, p_fr = probs.get("eng_Latn", 0.0), probs.get("fra_Latn", 0.0)
    if n_markers and max(p_en, p_fr) < 0.5:
        variety = "ary_latn"
    elif max(p_en, p_fr) >= 0.01:
        variety = "en" if p_en >= p_fr else "fr"
    else:
        variety = "ary_latn" if n_markers else "other"
    return variety, {"glotlid": top, "p": round(probs[top], 3), "markers": n_markers,
                     "p_en": round(p_en, 3), "p_fr": round(p_fr, 3)}


def evaluate(texts, gold):
    """Accuracy and confusion of identify() against known varieties."""
    import pandas as pd
    pred = [identify(t)[0] for t in texts]
    df = pd.DataFrame({"gold": list(gold), "pred": pred})
    return (df.gold == df.pred).mean(), pd.crosstab(df.gold, df.pred)
