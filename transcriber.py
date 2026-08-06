import difflib
import re
import unicodedata

import numpy as np

from data_loader import normalize_name

TARGET_RATE = 16000


def frames_to_float32(frames, sample_rate):
    if not frames:
        return None
    arr = np.concatenate(frames, axis=1)
    if arr.ndim > 1:
        arr = arr.mean(axis=0)
    mono = arr.astype(np.float32) / 32768.0
    if sample_rate and sample_rate != TARGET_RATE:
        idx = np.round(
            np.linspace(0, len(mono) - 1,
                        int(len(mono) * TARGET_RATE / sample_rate))
        ).astype(int)
        mono = mono[idx]
    return mono


def decode_audio_upload(data):
    try:
        import av
        import io
        container = av.open(io.BytesIO(data))
        stream = container.streams.audio[0]
        rate = stream.codec_context.sample_rate or 48000
        frames = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray())
        if not frames:
            return None
        return frames_to_float32(frames, rate)
    except Exception:
        return None


def transcribe(audio, model_name="small"):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        audio, language="it", vad_filter=True, beam_size=5
    )
    return " ".join(seg.text.strip() for seg in segments)


PARTICLES = {
    "di", "de", "da", "del", "della", "dei", "degli", "delle", "van", "der",
}

NUM_WORDS = {
    "uno": 1, "una": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5,
    "sei": 6, "sette": 7, "otto": 8, "nove": 9, "dieci": 10, "undici": 11,
    "dodici": 12, "tredici": 13, "quattordici": 14, "quindici": 15,
    "sedici": 16, "diciassette": 17, "diciotto": 18, "diciannove": 19,
    "venti": 20, "trenta": 30, "quaranta": 40, "cinquanta": 50,
    "sessanta": 60, "settanta": 70, "ottanta": 80, "novanta": 90,
    "ventuno": 21, "ventidue": 22, "ventitre": 23, "ventitré": 23,
    "ventiquattro": 24, "venticinque": 25, "ventisei": 26, "ventisette": 27,
    "ventotto": 28, "ventinove": 29,
    "trentuno": 31, "trentadue": 32, "trentatre": 33, "trentatré": 33,
    "trentaquattro": 34, "trentacinque": 35, "trentasei": 36, "trentasette": 37,
    "trentotto": 38, "trentanove": 39,
    "quarantuno": 41, "quarantadue": 42, "quarantatre": 43, "quarantatré": 43,
    "quarantaquattro": 44, "quarantacinque": 45, "quarantasei": 46,
    "quarantasette": 47, "quarantotto": 48, "quarantanove": 49,
    "cinquantuno": 51, "cinquantadue": 52, "cinquantatre": 53,
    "cinquantatré": 53, "cinquantaquattro": 54, "cinquantacinque": 55,
    "cinquantasei": 56, "cinquantasette": 57, "cinquantotto": 58,
    "cinquantanove": 59,
    "sessantuno": 61, "sessantadue": 62, "sessantatre": 63, "sessantatré": 63,
    "sessantaquattro": 64, "sessantacinque": 65, "sessantasei": 66,
    "sessantasette": 67, "sessantotto": 68, "sessantanove": 69,
    "settantuno": 71, "settantadue": 72, "settantatre": 73, "settantatré": 73,
    "settantaquattro": 74, "settantacinque": 75, "settantasei": 76,
    "settantasette": 77, "settantotto": 78, "settantanove": 79,
    "ottantuno": 81, "ottantadue": 82, "ottantatre": 83, "ottantatré": 83,
    "ottantaquattro": 84, "ottantacinque": 85, "ottantasei": 86,
    "ottantasette": 87, "ottantotto": 88, "ottantanove": 89,
    "novantuno": 91, "novantadue": 92, "novantatre": 93, "novantatré": 93,
    "novantaquattro": 94, "novantacinque": 95, "novantasei": 96,
    "novantasette": 97, "novantotto": 98, "novantanove": 99,
    "cento": 100, "duecento": 200, "trecento": 300, "quattrocento": 400,
    "cinquecento": 500, "seicento": 600, "mille": 1000,
}
_NUM_SORTED = sorted(NUM_WORDS, key=len, reverse=True)


def _words_to_number(text):
    low = text.lower().replace("'", " ")
    total = 0
    found = False
    i = 0
    while i < len(low):
        matched = False
        for word in _NUM_SORTED:
            if low.startswith(word, i):
                total += NUM_WORDS[word]
                i += len(word)
                found = True
                matched = True
                break
        if not matched:
            i += 1
    return total if found else None


def extract_bid(text, players_df):
    result = {"player": None, "price": None, "candidates": []}
    if not text:
        return result

    numbers = [int(n) for n in re.findall(r"\d+", text)]
    word_number = _words_to_number(text)
    if numbers:
        result["price"] = max(numbers)
    if word_number and (result["price"] is None or word_number > result["price"]):
        result["price"] = word_number

    clean = re.sub(r"[^\w\s'àèéìòù]", " ", text.lower())
    q = [w for w in clean.split() if len(w) > 1 and not w.isdigit()
         and w not in PARTICLES]
    q_single = [w for w in clean.split() if len(w) == 1 and w.isalpha()]
    if not q:
        return result
    q_joined = "".join(clean.split())

    scored = []
    for _, p in players_df.iterrows():
        raw_tokens = normalize_name(p["Nome"]).split()
        tokens = [t for t in raw_tokens if t not in PARTICLES]
        if not tokens:
            continue
        alias = "".join(raw_tokens)
        strong = len(alias) >= 5 and alias in q_joined
        if strong:
            score = 0.7
            tokens = [t for t in tokens if t in q]
            if tokens and all(t in q for t in tokens):
                score += 0.2
        else:
            common = set(q) & set(tokens)
            if not common:
                ratio = difflib.SequenceMatcher(
                    None, " ".join(q), " ".join(tokens)
                ).ratio()
                if ratio < 0.6:
                    continue
                score = ratio * 0.6
            else:
                score = 0.0
                if tokens[0] in q:
                    score += 0.5
                if len(tokens) > 1 and tokens[-1] in q:
                    score += 0.3
                score += 0.15 * max(0, len(common) - 1)
                if all(t in q for t in tokens):
                    score += 0.2
                if q_single and any(
                    tok.startswith(c) for tok in tokens[1:] for c in q_single
                ):
                    score += 0.15
        scored.append((score, p))

    scored.sort(key=lambda x: (x[0], x[1].get("FM", 0) or 0), reverse=True)
    top = [p for _, p in scored[:5] if _ >= 0.3]
    result["candidates"] = top
    if top:
        result["player"] = top[0]["Nome"]
    return result
