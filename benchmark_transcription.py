import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from data_loader import build_players
from transcriber import extract_bid

CLIP_DIR = Path(__file__).parent / "data" / "benchmark" / "clips"
TTS_DIR = Path("/tmp/fcp_tts")
MODELS = ["small", "medium", "large-v3-turbo"]

EXPECTED = {
    1: ("Thuram Marcus", 150),
    2: ("Dimarco Federico", 32),
    3: ("Martinez Lautaro", 108),
    4: ("Milinkovic Savic Vanja", 40),
    5: ("De Ketelaere Charles", 106),
    6: ("Lukaku Romelu", 200),
    7: ("Svilar Mile", 18),
    8: ("Malen Donyell", 100),
    9: ("Scamacca Gianluca", 97),
    10: ("Martinez Josep", 63),
}


def load_clips(folder):
    clips = {}
    for path in sorted(Path(folder).glob("*")):
        if path.suffix.lower() not in (".wav", ".aiff", ".aif", ".mp3"):
            continue
        try:
            data, rate = sf.read(str(path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if rate != 16000:
                idx = np.round(
                    np.linspace(0, len(data) - 1,
                                int(len(data) * 16000 / rate))
                ).astype(int)
                data = data[idx]
            key = int(path.stem.split("_")[0][:2])
            clips[key] = data
        except Exception:
            pass
    return clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--folder", default=str(CLIP_DIR))
    parser.add_argument("--beam", type=int, default=5)
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    clips = load_clips(args.folder)
    if not clips:
        print("Nessun clip trovato in", args.folder)
        return
    df = build_players()
    print(f"{len(clips)} clip da {args.folder}\n")

    results = {}
    for model_name in args.models:
        t0 = time.time()
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        load_s = time.time() - t0
        ok_p = ok_n = 0
        lat = []
        rows = []
        for key in sorted(clips):
            audio = clips[key]
            t0 = time.time()
            segments, _ = model.transcribe(
                audio, language="it", vad_filter=True, beam_size=args.beam
            )
            text = " ".join(s.text.strip() for s in segments)
            lat.append(time.time() - t0)
            r = extract_bid(text, df)
            exp_p, exp_price = EXPECTED[key]
            p_ok = r["player"] == exp_p
            n_ok = r["price"] == exp_price
            ok_p += p_ok
            ok_n += n_ok
            rows.append((key, text, r, p_ok, n_ok))
        results[model_name] = (ok_p, ok_n, lat, rows)
        print(f"--- {model_name} (load {load_s:.0f}s, "
              f"avg {np.mean(lat):.2f}s/clip) ---")
        for key, text, r, p_ok, n_ok in rows:
            exp_p, exp_price = EXPECTED[key]
            print(f"  {key:02d} {text!r:48} -> {str(r['player']):26} "
                  f"{r['price'] if r['price'] is not None else '-':>4}  "
                  f"{'OK' if p_ok else 'MISS'} "
                  f"{'OK' if n_ok else 'MISS'}  (atteso {exp_p}, {exp_price})")
        print(f"  => player {ok_p}/{len(clips)} · price {ok_n}/{len(clips)}"
              f" · lat media {np.mean(lat):.2f}s\n")

    print("=== RIEPILOGO ===")
    print(f"{'modello':18} {'player':8} {'price':8} {'lat media':10}")
    for name, (ok_p, ok_n, lat, _) in results.items():
        print(f"{name:18} {ok_p}/{len(clips):<4} {ok_n}/{len(clips):<5} "
              f"{np.mean(lat):.2f}s")


if __name__ == "__main__":
    main()
