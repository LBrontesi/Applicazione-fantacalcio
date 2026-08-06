import time

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer

import asta_core
import data_loader
import scraper
from data_loader import ROLE_ORDER, build_players, fair_values_scaled, suggest_player
from transcriber import (
    decode_audio_upload, extract_bid, frames_to_float32, transcribe_audio,
)

st.set_page_config(page_title="Asta Coach", page_icon="⚽", layout="wide")

MAX_REC_SECONDS = 15
SAMPLE_RATE = 16000
LIVE_MODEL = "small"
FINAL_MODEL = "large-v3-turbo"
LIVE_CHUNK_SECONDS = 2.5
DEBUG_LOG = data_loader.DATA_DIR / "webrtc_debug.log"


def _debug_log(msg):
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(f"{time.time():.1f} {msg}\n")
    except Exception:
        pass


class BidRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames = []
        self.rate = 48000
        self.peak = 0.0
        self._log_once = False

    def _collect(self, frame):
        arr = frame.to_ndarray().copy()
        self.frames.append(arr)
        self.rate = frame.sample_rate
        if not self._log_once:
            self._log_once = True
            _debug_log(f"first frame: rate={self.rate} shape={arr.shape} "
                       f"dtype={arr.dtype}")
        if arr.size:
            self.peak = max(self.peak, float(np.abs(arr).max()))

    def recv(self, frame):
        self._collect(frame)
        return frame

    async def recv_queued(self, frames):
        for frame in frames:
            self._collect(frame)
        return frames


def recorder_factory():
    proc = BidRecorder()
    st.session_state["rec_proc"] = proc
    return proc


@st.cache_resource
def get_whisper(model_name):
    from faster_whisper import WhisperModel
    return WhisperModel(model_name, device="cpu", compute_type="int8")


@st.cache_data(show_spinner=False)
def load_players():
    return build_players()


def players_df():
    df = load_players()
    if df.empty:
        return df
    return df


def save_session_state():
    session = st.session_state.get("session")
    if session:
        asta_core.save_session(session)


def render_setup():
    st.header("⚙️ Setup")
    st.subheader("🔄 Dati")
    c1, c2 = st.columns(2)
    if c1.button("Scarica quotazioni Gazzetta (veloce)"):
        with st.status("Scaricando quotazioni...") as status:
            scraper.scrape_quotazioni(progress_cb=lambda m: status.update(label=m))
            st.cache_data.clear()
            status.update(label="Fatto!", state="complete")
        st.rerun()
    if c2.button("Scarica lista giocatori FCP (lento, ~10 min)"):
        with st.status("Scaricando giocatori da fantacalciopedia...") as status:
            scraper.scrape_fantacalciopedia(
                progress_cb=lambda m: status.update(label=m)
            )
            st.cache_data.clear()
            status.update(label="Fatto!", state="complete")
        st.rerun()

    df = players_df()
    if not df.empty:
        st.caption(
            f"{len(df)} giocatori · {len(df[df['QA'].notna()])} con quotazione · "
            f"{len(df[df['FM'].notna()])} con fantamedia"
        )
        with st.expander("Anteprima dati"):
            st.dataframe(
                df[
                    ["Nome", "Squadra", "Ruolo", "FM", "QA", "QI", "Cluster",
                     "Attributi", "ResInf"]
                ].head(50),
                width="stretch",
            )

    st.divider()
    st.subheader("🏟️ Sessione")
    sessions = asta_core.list_sessions()
    if sessions:
        labels = {str(p): p.stem.replace("asta_", "") for p in sessions}
        pick = st.selectbox("Riprendi sessione", [""] + list(labels))
        if pick and st.button("Carica"):
            session = asta_core.load_session(pick)
            st.session_state["session"] = session
            my_team = session["meta"].get("my_team")
            if my_team in session["teams"]:
                st.session_state["my_team"] = my_team
            st.rerun()

    st.markdown("**Nuova sessione**")
    league = st.text_input("Nome lega", value="Asta 26-27")
    teams_input = st.text_area(
        "Squadre (una per riga)",
        value="Bro\nLolo\nGiorgio\nRochira\nPistacchio\nBugani\nRiolo\nBabbo\n"
              "Piermattei\nFrancesco",
        height=140,
    )
    budget = st.number_input("Budget iniziale", min_value=10, max_value=5000,
                             value=500, step=10)
    slots = {}
    sc = st.columns(4)
    for i, role in enumerate(ROLE_ORDER):
        slots[role] = sc[i].number_input(
            f"Slot {role}", min_value=1, max_value=12,
            value=data_loader.DEFAULT_SLOTS[role],
        )

    fair_default = fair_values_scaled(budget)
    fair_rows = []
    for role in ROLE_ORDER:
        for slot, value in enumerate(fair_default[role], start=1):
            fair_rows.append({"Ruolo": role, "Slot": slot, "Fair Value": value})
    fair_df = pd.DataFrame(fair_rows)
    st.markdown("**Tabella fair value (per slot)**")
    edited = st.data_editor(
        fair_df, num_rows="fixed", width="stretch", key="fair_editor"
    )
    fair = {}
    for role in ROLE_ORDER:
        fair[role] = edited[edited["Ruolo"] == role]["Fair Value"] \
            .astype(int).tolist()

    if st.button("Crea sessione", type="primary"):
        teams = [t.strip() for t in teams_input.splitlines() if t.strip()]
        if len(teams) != 10:
            st.error(f"La lega è a 10 squadre — ora ne hai inserite {len(teams)}")
        else:
            session = asta_core.new_session(
                league, teams, budget=int(budget), slots=slots, fair=fair
            )
            session["meta"]["my_team"] = teams[0]
            st.session_state["session"] = session
            st.session_state["my_team"] = teams[0]
            asta_core.save_session(session)
            st.session_state["tab_target"] = "Asta Live"
            st.rerun()


def player_card(player, info):
    c = st.columns(6)
    c[0].metric("Squadra", player["Squadra"])
    c[1].metric("Ruolo", player["Ruolo"])
    c[2].metric("FantaMedia", f"{player['FM']:.2f}" if pd.notna(player["FM"]) else "-")
    qa = player["QA"]
    c[3].metric("Quotazione QA", f"{qa:.0f}" if pd.notna(qa) else "-")
    c[4].metric("Cluster", f"{info['cluster']}")
    c[5].metric("Fair value slot", info["cap"])

    if info["full"]:
        st.error(f"Rosa completa per il ruolo {player['Ruolo']} — non puoi più "
                 f"comprare {ROLE_ORDER}")
    else:
        st.markdown(
            f"Bid fino a **{info['cap']}** crediti "
            f"(slot {info['next_free']}/{info['slots']} libero · cluster "
            f"{info['cluster']})"
        )
        st.caption(
            f"Bid utili: apertura suggerita {qa * session_budget() // 100} · "
            f"riserva attuale: {asta_core.reserve(st.session_state['session'])}"
        )


def session_budget():
    session = st.session_state.get("session")
    return session["meta"]["budget"] if session else 500


@st.fragment(run_every="2s")
def live_loop():
    if not st.session_state.get("rec_on"):
        return
    proc = st.session_state.get("rec_proc")
    if proc is None:
        st.markdown(
            '<div style="background:#7a1f1f;color:white;padding:10px 14px;'
            'border-radius:8px;font-weight:bold">🔴 REC — in attesa del '
            'microfono…</div>',
            unsafe_allow_html=True,
        )
        return
    n = len(proc.frames)
    peak = proc.peak
    peak_norm = peak / 32768.0 if peak > 1.0 else peak
    elapsed = int(time.time() - st.session_state.get("rec_start", time.time()))
    _debug_log(f"live tick: frames={n} peak={peak_norm:.3f} "
               f"base={st.session_state.get('chunk_base', 0)}")
    st.markdown(
        f'<div style="background:#7a1f1f;color:white;padding:10px 14px;'
        f'border-radius:8px;font-weight:bold">🔴 REC — ascoltando '
        f'({elapsed}s · {n} frame · livello {peak_norm:.2f}) — parla '
        f'chiaro e premi **Ferma**</div>',
        unsafe_allow_html=True,
    )
    st.progress(min(max(peak_norm * 3, 0.0), 1.0), text="livello microfono")
    if elapsed >= 3 and n == 0:
        st.warning("Nessun audio ricevuto: controlla il permesso del "
                   "microfono (🔒 nella barra indirizzi) e riprova.")
    if elapsed >= MAX_REC_SECONDS:
        st.session_state["rec_on"] = False
        st.session_state["rec_autostop"] = True
        st.rerun()
    base = st.session_state.get("chunk_base", 0)
    if len(proc.frames) <= base:
        return
    new = proc.frames[base:]
    st.session_state["chunk_base"] = len(proc.frames)
    audio = frames_to_float32(new, proc.rate)
    if audio is None:
        return
    model = get_whisper(LIVE_MODEL)
    text = transcribe_audio(audio, model).strip()
    if text:
        st.session_state["live_text"] = (
            st.session_state.get("live_text", "") + " " + text
        ).strip()
    live_text = st.session_state.get("live_text", "")
    placeholder = "… in attesa di parole …"
    st.markdown(
        f'<div style="background:#1f2a44;color:#cfe3ff;padding:10px 14px;'
        f'border-radius:8px;min-height:44px">'
        f'{live_text or placeholder}</div>',
        unsafe_allow_html=True,
    )
    r = extract_bid(live_text, players_df())
    if r["price"]:
        who = f" · {r['player']} (probabile)" if r["player"] else ""
        st.success(f"💰 Prezzo rilevato dal vivo: **{r['price']}**{who}")
    st.caption("trascrizione live (modello veloce) — alla fine usa quello "
               "ad alta precisione")


def set_prefill(key, value):
    st.session_state[f"{key}_value"] = value
    st.session_state.pop(key, None)


def sync_widget(key):
    if key in st.session_state:
        st.session_state[f"{key}_value"] = st.session_state[key]


def render_recorder():
    st.markdown("**🎙️ Trascrittore**")
    rec_on = st.session_state.get("rec_on", False)

    b1, b2 = st.columns(2)
    if not rec_on:
        if b1.button("🎙️ Inizia registrazione", type="primary"):
            st.session_state["rec_on"] = True
            st.session_state["rec_start"] = time.time()
            st.session_state["rec_empty_warn"] = False
            st.rerun()
    else:
        if b1.button("⏹️ Ferma e trascrivi", type="primary"):
            st.session_state["rec_on"] = False
            st.rerun()

    try:
        ctx = webrtc_streamer(
            key="bid_rec",
            mode=WebRtcMode.SENDONLY,
            desired_playing_state=rec_on,
            audio_processor_factory=recorder_factory,
            audio_receiver_size=1024,
            async_processing=True,
            media_toggle_controls=False,
            media_stream_constraints={"audio": True, "video": False},
        )
    except Exception:
        ctx = None

    proc = st.session_state.get("rec_proc")
    if ctx is not None:
        live = ctx.audio_processor
        if live is not None:
            proc = live
            st.session_state["rec_proc"] = live

    if rec_on:
        live_loop()
        if st.session_state.pop("rec_autostop", False):
            st.info("Registrazione fermata automaticamente dopo "
                    f"{MAX_REC_SECONDS}s")
    elif proc is not None and proc.frames:
        audio = frames_to_float32(proc.frames, proc.rate)
        peak = proc.peak
        proc.frames.clear()
        proc.peak = 0.0
        st.session_state["rec_proc"] = None
        st.session_state["live_text"] = ""
        st.session_state["chunk_base"] = 0
        if audio is not None and len(audio) > 0 and peak > 0:
            st.session_state["recorded_audio"] = audio
        else:
            st.session_state["rec_empty_warn"] = True

    if st.session_state.pop("rec_empty_warn", False):
        st.warning("Nessun audio ricevuto dal microfono — controlla che il "
                   "browser abbia il permesso (icona 🔒 nella barra degli "
                   "indirizzi) e riprova, oppure usa l'upload qui sotto.")

    if st.session_state.get("recorded_audio") is not None:
        with st.spinner("Trascrivendo audio (modello ad alta precisione)..."):
            text = transcribe_audio(
                st.session_state["recorded_audio"], get_whisper(FINAL_MODEL)
            )
        st.session_state["recorded_audio"] = None
        if text.strip():
            st.session_state["transcript_area"] = text
            st.session_state["last_final"] = text
        else:
            st.session_state["rec_empty_warn"] = True
        st.rerun()

    st.caption("Inizia → parla → Ferma: il testo scorre dal vivo e alla fine "
               "la trascrizione usa il modello ad alta precisione.")

    uploaded = st.file_uploader(
        "oppure carica un file audio", type=["wav", "mp3", "m4a", "aac",
                                             "ogg", "flac"],
    )
    if uploaded is not None \
            and st.session_state.get("uploaded_name") != uploaded.name:
        audio = decode_audio_upload(uploaded.getvalue())
        if audio is not None:
            st.session_state["recorded_audio"] = audio
            st.session_state["uploaded_name"] = uploaded.name
            st.rerun()
        else:
            st.error("File audio non decodificabile")

    transcript = st.session_state.get("transcript_area", "")
    st.text_area("Trascrizione", key="transcript_area", height=80)
    last_final = st.session_state.get("last_final", "")
    if last_final and not transcript:
        st.caption("(ultima trascrizione)")
    if transcript and st.button("🔍 Estrai bid dalla trascrizione"):
        res = extract_bid(transcript, players_df())
        if res["player"]:
            set_prefill("nom_search", res["player"])
        if res["price"]:
            set_prefill("nom_price", res["price"])
        if res["candidates"]:
            st.session_state["nom_alt"] = [p["Nome"] for p in res["candidates"]]
        st.rerun()


def render_asta():
    session = st.session_state.get("session")
    if not session:
        st.info("Crea o riprendi una sessione nella tab Setup")
        return

    players = players_df()
    if players.empty:
        st.warning("Nessun dato giocatori — scarica le quotazioni nella tab Setup")
        return

    st.header("🎯 Asta Live")
    my_team = session["meta"].get("my_team") or st.session_state.get("my_team")
    if my_team not in session["teams"]:
        my_team = list(session["teams"])[0]

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Nominazione")
        st.text_input(
            "Cerca giocatore", key="nom_search",
            value=st.session_state.get("nom_search_value", ""),
            on_change=lambda: sync_widget("nom_search"),
        )
        q = st.session_state.get("nom_search", "")
        suggestions = suggest_player(q, players) if len(q) >= 2 else pd.DataFrame()
        if len(q) >= 2 and suggestions.empty:
            st.caption("Nessun match, prova altro nome")
        if not suggestions.empty:
            choice = st.selectbox(
                "Giocatore selezionato",
                suggestions["Nome"].tolist(),
                format_func=lambda n: (
                    f"{n} · {suggestions[suggestions['Nome'] == n].iloc[0]['Squadra']} "
                    f"({suggestions[suggestions['Nome'] == n].iloc[0]['Ruolo']})"
                ),
                key="nom_select",
            )
            player = suggestions[suggestions["Nome"] == choice].iloc[0]
            info = asta_core.coach(session, player)
            player_card(player, info)

            alt = st.session_state.get("nom_alt", [])
            if alt and len(alt) > 1:
                st.caption(f"Possibili: {', '.join(alt)}")

            st.markdown("**Registra aggiudicazione**")
            f1, f2, f3 = st.columns(3)
            buyers = list(session["teams"].keys())
            buyer = f1.selectbox("Aggiudicatario", buyers, key="bid_buyer")
            price = f2.number_input(
                "Prezzo (crediti)", min_value=1, max_value=session["meta"]["budget"],
                value=int(st.session_state.get("nom_price_value", 1)), step=1,
                key="nom_price",
                on_change=lambda: sync_widget("nom_price"),
            )
            note = f3.text_input("Nota", key="nom_note")
            budget_left = session["teams"][buyer]["budget_left"]
            if price > budget_left:
                st.warning(f"{buyer} ha solo {budget_left} crediti rimasti!")
            if info["full"]:
                st.error(f"⛔ Rosa completa per il ruolo {player['Ruolo']} — "
                         f"non puoi più comprare")
            elif price > info["cap"]:
                st.error(
                    f"⛔ **FERMATI — NON OFFRIRE PIÙ!**\n\n"
                    f"{player['Nome']} (cluster {info['cluster']}) ha "
                    f"superato il cap: offerta **{price}** > cap **{info['cap']}**. "
                    f"Esci dall'asta, non vale più la pena."
                )
            else:
                st.success(
                    f"✅ Offerta {price} entro il cap di {info['cap']} — "
                    f"puoi spingere fino a **{info['cap']}**"
                )

            if st.button("✔️ Registra vendita", type="primary"):
                asta_core.register_bid(
                    session, player["Nome"], player["Ruolo"],
                    int(player["Cluster"]), buyer, int(price), int(info["cap"]),
                    note=note, source="transcript" if st.session_state.get(
                        "nom_alt") else "manual",
                )
                asta_core.save_session(session)
                for k in ["nom_search", "nom_alt"]:
                    st.session_state.pop(k, None)
                st.session_state["transcript_area"] = ""
                st.rerun()
            must_have = session["teams"][my_team].get("must_have", [])
            if player["Nome"] not in must_have:
                if st.button("⭐ Aggiungi ai must-have"):
                    must_have.append(player["Nome"])
                    session["teams"][my_team]["must_have"] = must_have
                    asta_core.save_session(session)
                    st.rerun()
            else:
                if st.button("Rimuovi dai must-have"):
                    must_have.remove(player["Nome"])
                    session["teams"][my_team]["must_have"] = must_have
                    asta_core.save_session(session)
                    st.rerun()

        st.divider()
        render_recorder()

        st.subheader("📜 Log vendite")
        if session["events"]:
            log = pd.DataFrame(session["events"])
            st.dataframe(
                log[["ts", "player", "buyer", "price", "cap", "note", "source"]],
                width="stretch", hide_index=True,
            )
        if st.button("↩️ Annulla ultima vendita"):
            last = asta_core.undo(session)
            asta_core.save_session(session)
            if last:
                st.success(f"Annullata: {last['player']} → {last['buyer']} "
                           f"({last['price']})")
            st.rerun()

    with right:
        st.subheader(f"👤 La tua squadra: {my_team}")
        me = session["teams"][my_team]
        c1, c2, c3 = st.columns(3)
        c1.metric("Budget", me["budget_left"])
        c2.metric("Riserva", asta_core.reserve(session))
        c3.metric("Giocatori", len(me["roster"]))
        for role in ROLE_ORDER:
            n = len([i for i in me["roster"] if i["role"] == role])
            max_n = session["meta"]["slots"][role]
            color = "red" if n >= max_n else "green"
            st.markdown(
                f"{role}: **{n}/{max_n}** "
                f"{'<span style=\"color:red\">FULL</span>' if n >= max_n else ''}",
                unsafe_allow_html=True,
            )
        if me["roster"]:
            roster_df = pd.DataFrame(me["roster"])
            st.dataframe(
                roster_df[["player", "cluster", "price", "cap", "note"]],
                width="stretch", hide_index=True,
            )
        st.caption("Must-have:")
        for name in me.get("must_have", []):
            sold = any(e["player"] == name for e in session["events"])
            status = "✅ comprato" if sold else "🆓 ancora in lista"
            st.markdown(f"- {name} — {status}")

        st.divider()
        st.subheader("👀 Mercato rivali")
        for row in asta_core.market_watch(session, exclude=my_team):
            counts = " · ".join(
                f"{r}: {row['counts'][r]}" for r in ROLE_ORDER
            )
            st.markdown(
                f"**{row['team']}** (budget {row['budget_left']}): {counts}"
            )


def render_summary():
    session = st.session_state.get("session")
    if not session:
        st.info("Crea o riprendi una sessione nella tab Setup")
        return
    st.header("📊 Riepilogo")
    rows = []
    for team, data in session["teams"].items():
        spent = session["meta"]["budget"] - data["budget_left"]
        rows.append(
            {"Squadra": team, "Giocatori": len(data["roster"]),
             "Speso": spent, "Budget rimasto": data["budget_left"]}
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    for team, data in session["teams"].items():
        with st.expander(f"{team} — {len(data['roster'])} giocatori"):
            if data["roster"]:
                st.dataframe(
                    pd.DataFrame(data["roster"])
                    .sort_values("role")[["player", "role", "cluster", "price",
                                          "cap", "note"]],
                    width="stretch", hide_index=True,
                )
            else:
                st.caption("Nessun giocatore")

    all_rows = []
    for team, data in session["teams"].items():
        for item in data["roster"]:
            all_rows.append({**item, "squadra_asta": team})
    if all_rows:
        export = pd.DataFrame(all_rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Esporta CSV", export,
            file_name=f"{session['meta']['league']}_asta.csv",
            mime="text/csv",
        )


def main():
    st.sidebar.title("⚽ Asta Coach")
    target = st.session_state.pop("tab_target", None)
    if target:
        st.session_state["main_tabs"] = target
    session = st.session_state.get("session")
    if session:
        my_team = st.session_state.get("my_team")
        teams = list(session["teams"].keys())
        idx = teams.index(my_team) if my_team in teams else 0
        st.sidebar.selectbox(
            "La tua squadra", teams, index=idx, key="my_team",
            on_change=lambda: (
                session["meta"].__setitem__(
                    "my_team", st.session_state.get("my_team", teams[0])),
                save_session_state(),
            ),
        )
        st.sidebar.caption(
            f"{session['meta']['league']} · budget {session['meta']['budget']}"
        )
        if st.sidebar.button("💾 Salva sessione"):
            asta_core.save_session(session)
            st.sidebar.success("Salvata")

    tab_setup, tab_asta, tab_summary = st.tabs(
        ["Setup", "Asta Live", "Riepilogo"], key="main_tabs"
    )
    with tab_setup:
        render_setup()
    with tab_asta:
        render_asta()
    with tab_summary:
        render_summary()


if __name__ == "__main__":
    main()
