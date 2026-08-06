import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

CLIP_DIR = Path(__file__).parent / "data" / "benchmark" / "clips"
SAMPLE_RATE = 16000
SECONDS = 5

PHRASES = [
    ("Offro centocinquanta per Thuram", "Thuram Marcus", 150),
    ("Dimarco a trentadue", "Dimarco Federico", 32),
    ("Lautaro a centootto", "Martinez Lautaro", 108),
    ("vendo Milinkovic a quaranta", "Milinkovic Savic Vanja", 40),
    ("De Ketelaere per centosei", "De Ketelaere Charles", 106),
    ("Lukaku a duecento", "Lukaku Romelu", 200),
    ("Svilar a diciotto", "Svilar Mile", 18),
    ("Malen a cento", "Malen Donyell", 100),
    ("Scamacca a novantasette", "Scamacca Gianluca", 97),
    ("Martinez Jo a sessantatre", "Martinez Josep", 63),
]


def main():
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    print("Microfono: premi INVIO, leggi la frase ad alta voce, poi resta "
          "in silenzio.")
    print(f"Ogni clip dura {SECONDS}s. Puoi rifare una clip rispondendo y.\n")
    for i, (phrase, player, price) in enumerate(PHRASES, start=1):
        while True:
            input(f"[{i}/10] PREMI INVIO e di':  «{phrase}»  ")
            audio = sd.rec(int(SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                           channels=1, dtype="float32")
            sd.wait()
            peak = float(np.abs(audio).max())
            if peak < 0.02:
                print(f"    livello molto basso ({peak:.2f}) — parla più "
                      f"vicino al microfono")
            else:
                print(f"    registrato (livello {peak:.2f})")
            redo = input("    ok? [Invio=ok, y=riprova] ").strip().lower()
            if redo != "y":
                break
        path = CLIP_DIR / f"{i:02d}.wav"
        sf.write(str(path), audio, SAMPLE_RATE)
        print(f"    salvato {path.name}\n")
    print("Fatto! Ora puoi lanciare python benchmark_transcription.py")


if __name__ == "__main__":
    main()
