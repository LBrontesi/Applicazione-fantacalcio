from datetime import datetime

import pandas as pd
import streamlit as st

import asta_core
import data_loader
import scraper
from data_loader import (
    ROLE_ORDER, build_lineups, build_players, fair_values_scaled,
    load_ranking_weights, save_ranking_weights, suggest_player,
    backtest_predictor, DEFAULT_METHOD,
)
from scraper import ScrapeError

WEIGHT_LABELS = {
    "FM": "Fantamedia (FCP)",
    "FVM": "Fantamedia (Gazzetta)",
    "ALG": "Algoritmo (FCP)",
    "Starter": "Probabile titolare",
    "SetPieces": "Set-piece (rigorista/angoli/punizioni)",
    "Tags": "Attributi (Fuoriclasse, Goleador, …)",
    "Injury": "Robustezza infortuni (premia chi non si infortuna)",
    "Availability": "Affidabilità d'impiego (formazione + robustezza)",
    "ExpectedOutput": "xG + xA per 90 (CSV storico)",
}

ROLE_PLAN_CHOICES = {
    "Risparmia": 0.8,
    "Equilibrio": 1.0,
    "Spingi": 1.2,
}

WATCHLIST_EXPLANATIONS = {
    "A": "Obiettivo: puoi spingere fino al prezzo consigliato.",
    "B": "Alternativa: segui il prezzo consigliato senza inseguire.",
    "C": "Occasione: punta solo se il prezzo resta conveniente.",
}

st.set_page_config(page_title="Asta Coach", page_icon="⚽", layout="wide")

# Cambiare quando cambia lo schema restituito da build_players: evita che una
# cache Streamlit della versione precedente venga riutilizzata dalla nuova UI.
PLAYER_CACHE_SCHEMA_VERSION = 2


def inject_styles():
    st.markdown(
        """
        <style>
        .app-hero {
            background: linear-gradient(135deg, #12372a 0%, #1f5c2e 100%);
            border-radius: 14px;
            color: #f4fbf5;
            padding: 20px 24px;
            margin: 0 0 18px;
        }
        .app-hero h1 {
            font-size: 2rem;
            line-height: 1.1;
            margin: 0 0 6px;
        }
        .app-hero p {
            color: #d7eadb;
            margin: 0;
        }
        .bid-card {
            align-items: center;
            background: linear-gradient(135deg, #174b2d 0%, #287a3d 100%);
            border-radius: 14px;
            color: white;
            display: flex;
            justify-content: space-between;
            gap: 18px;
            margin: 8px 0 16px;
            padding: 18px 22px;
        }
        .bid-label {
            color: #cfe8d4;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .bid-value {
            font-size: 2rem;
            font-weight: 750;
            line-height: 1.1;
            margin-top: 4px;
        }
        .bid-meta {
            color: #d7eadb;
            font-size: 0.9rem;
            margin-top: 5px;
        }
        .bid-stop {
            border: 1px solid rgba(255, 255, 255, 0.35);
            border-radius: 999px;
            color: #f2fff4;
            font-size: 0.85rem;
            padding: 8px 12px;
            text-align: center;
            white-space: nowrap;
        }
        .section-note {
            color: #52635a;
            font-size: 0.92rem;
            margin: -8px 0 14px;
        }
        @media (max-width: 768px) {
            .app-hero { padding: 16px 18px; }
            .app-hero h1 { font-size: 1.55rem; }
            .bid-card {
                align-items: flex-start;
                flex-direction: column;
                gap: 10px;
                padding: 16px 18px;
            }
            .bid-value { font-size: 1.75rem; }
            .bid-stop { white-space: normal; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_scrape(jobs):
    with st.status("Scaricando dati...") as status:
        try:
            for label, job in jobs:
                status.update(label=label)
                job()
            st.cache_data.clear()
            status.update(label="Fatto!", state="complete")
        except ScrapeError as exc:
            status.update(label="Errore", state="error")
            st.error(f"Download non riuscito: {exc}")
        except Exception as exc:
            status.update(label="Errore", state="error")
            st.error(f"Errore inatteso: {exc}")


@st.cache_data(show_spinner=False)
def load_players(schema_version=PLAYER_CACHE_SCHEMA_VERSION):
    del schema_version
    return build_players()


@st.cache_data(show_spinner=False)
def load_lineups():
    players = build_players()
    return build_lineups(players)


def load_excluded():
    path = scraper.DATA_DIR / "esclusi.csv"
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["Nome"])


def save_excluded(names):
    path = scraper.DATA_DIR / "esclusi.csv"
    scraper.DATA_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"Nome": sorted(names)}).to_csv(path, index=False)


def players_df():
    return load_players()


def get_config():
    session = st.session_state.get("session")
    if session:
        asta_core.ensure_session(session)
        return session["meta"]["budget"], session["meta"]["fair"]
    return 500, fair_values_scaled(500)


def closest_role_plan(value):
    return min(ROLE_PLAN_CHOICES, key=lambda label: abs(
        ROLE_PLAN_CHOICES[label] - float(value)
    ))


def source_freshness():
    """Return short, auction-day friendly freshness labels for local sources."""
    sources = [
        ("Quotazioni", scraper.QUOTAZIONI, 24),
        ("Giocatori/FCP", scraper.PLAYERS_FCP, 24 * 7),
        ("Formazioni", scraper.FORMAZIONI, 8),
        ("Tiratori", scraper.SET_PIECES, 24 * 7),
    ]
    now = datetime.now().timestamp()
    result = []
    for label, path, fresh_hours in sources:
        if not path.exists():
            result.append({"label": label, "value": "Manca", "state": "🔴"})
            continue
        age_hours = max(0, (now - path.stat().st_mtime) / 3600)
        if age_hours < 1:
            value = f"{max(1, round(age_hours * 60))} min fa"
        elif age_hours < 24:
            value = f"{age_hours:.0f} ore fa"
        else:
            value = f"{age_hours / 24:.0f} giorni fa"
        state = "🟢" if age_hours <= fresh_hours else "🟠"
        result.append({"label": label, "value": value, "state": state})
    return result


def roster_alerts(own, players, slots):
    """Return only actionable roster risks; avoid warnings at auction start."""
    purchases = own.get("purchases", [])
    if not purchases:
        return []
    alerts = []
    clubs = pd.Series([
        str(item.get("club", "")).strip() for item in purchases
        if str(item.get("club", "")).strip()
    ]).value_counts()
    crowded = [f"{club} ({count})" for club, count in clubs.items() if count >= 3]
    if crowded:
        alerts.append("Troppi giocatori della stessa squadra: " + ", ".join(crowded) + ".")

    total_spent = max(1, int(own.get("spent", 0)))
    for role in ROLE_ORDER:
        role_spent = sum(int(item.get("price", 0)) for item in purchases
                         if item.get("role") == role)
        if len(purchases) >= 5 and role_spent / total_spent >= 0.45:
            alerts.append(
                f"{role} assorbe il {role_spent / total_spent:.0%} dei crediti spesi."
            )

    if len(purchases) >= 10:
        for role in ROLE_ORDER:
            planned = max(1, int(slots.get(role, 0)))
            bought = int(own.get("by_role", {}).get(role, 0))
            if bought == 0 or (planned >= 4 and bought / planned < 0.25):
                alerts.append(f"Reparto {role} molto indietro: {bought}/{planned} giocatori.")

    names = {str(item.get("name", "")) for item in purchases}
    owned = players[players["Nome"].isin(names)]
    if len(purchases) >= 8 and not owned.empty:
        starters = int(owned["Starter"].fillna(0).sum())
        if starters < max(2, round(len(purchases) * 0.40)):
            alerts.append(f"Pochi titolari probabili in rosa: {starters}/{len(purchases)}.")
        attacking = owned[owned["Ruolo"].isin(["C", "A"])]
        if len(attacking) >= 3 and not (attacking["SetPieces"].fillna(0) >= 0.3).any():
            alerts.append("Nessun centrocampista/attaccante con piazzati rilevati.")
    return alerts


def render_setup():
    st.header("⚙️ Setup")
    st.markdown(
        '<p class="section-note">Aggiorna le fonti, definisci il budget e '
        'controlla il metodo con cui vengono costruiti rank e cluster.</p>',
        unsafe_allow_html=True,
    )
    st.subheader("🔄 Dati")
    advanced_present = scraper.ADVANCED_STATS.exists()
    st.caption(
        "Un solo aggiornamento per arrivare pronto all'asta: quotazioni, lista "
        "giocatori, formazioni, indisponibili e tiratori. Le statistiche avanzate "
        f"CSV vengono {'mantenute' if advanced_present else 'lasciate vuote'} e non sovrascritte."
    )
    if st.button("🚀 Aggiorna tutto per l'asta (~10 min)", type="primary",
                 use_container_width=True):
        run_scrape([
            ("1/4 Scaricando quotazioni Gazzetta...",
             lambda: scraper.scrape_quotazioni(progress_cb=None)),
            ("2/4 Scaricando lista giocatori FCP (può richiedere alcuni minuti)...",
             lambda: scraper.scrape_fantacalciopedia(progress_cb=None)),
            ("3/4 Aggiornando formazioni, ballottaggi e indisponibili...",
             lambda: scraper.scrape_lineups(progress_cb=None)),
            ("4/4 Aggiornando rigoristi e tiratori...",
             lambda: scraper.scrape_set_pieces(progress_cb=None)),
        ])
        st.rerun()
    st.caption(
        "Dopo l'aggiornamento vengono ricalcolati automaticamente ranking, "
        "cluster, valore stagione e suggerimenti per l'asta live."
    )

    with st.expander("📥 Statistiche storiche avanzate (consigliato)"):
        st.caption(
            "Esporta le statistiche giocatori Serie A da FBref o FotMob in CSV e "
            "caricale qui. L'app riconosce automaticamente nome, squadra, presenze, "
            "titolarità, minuti, xG, xA, gol, assist, cartellini e gare saltate. È un import locale: "
            "non dipende da API fragili il giorno dell'asta."
        )
        advanced_file = st.file_uploader(
            "CSV FBref / FotMob", type=["csv"], key="advanced_stats_upload"
        )
        if st.button("Importa statistiche avanzate", disabled=advanced_file is None):
            try:
                imported = scraper.import_advanced_stats(advanced_file)
                st.cache_data.clear()
                st.success(f"Importati {len(imported)} giocatori. Ranking aggiornato.")
                st.rerun()
            except ScrapeError as exc:
                st.error(str(exc))

    df = players_df()
    if not df.empty:
        summary = st.columns(5)
        summary[0].metric("Giocatori", len(df))
        summary[1].metric("Con quotazione", int(df["QA"].notna().sum()))
        summary[2].metric("Con fantamedia", int(df["FM"].notna().sum()))
        summary[3].metric("Titolari probabili", int(df["Starter"].sum()))
        summary[4].metric("Con minuti storici", int(df["Minuti"].notna().sum()))
        with st.expander("Anteprima dati"):
            st.dataframe(
                df[
                    ["Nome", "Squadra", "Ruolo", "FM", "QA", "QI", "Cluster",
                     "Attributi", "ResInf"]
                ].head(50),
            )

    st.divider()
    st.subheader("⚙️ Configurazione cap (fair value per cluster)")
    st.caption(
        "Ogni ruolo è diviso in cluster da 10 giocatori (per fantamedia, "
        "uno per squadra). Il cap di un giocatore è il fair value del suo "
        "cluster: oltre quel prezzo conviene lasciar perdere."
    )
    session_meta = st.session_state.get("session", {}).get("meta", {})
    st.session_state.setdefault("cfg_budget", int(session_meta.get("budget", 500)))
    budget = st.number_input("Budget iniziale", min_value=10, max_value=5000,
                             step=10, key="cfg_budget")
    fair_default = st.session_state.pop("_loaded_fair", None)
    if not fair_default:
        fair_default = fair_values_scaled(int(budget))
    fair_rows = []
    for role in ROLE_ORDER:
        for slot, value in enumerate(fair_default[role], start=1):
            fair_rows.append({"Ruolo": role, "Slot": slot, "Fair Value": value})
    fair_df = pd.DataFrame(fair_rows)
    edited = st.data_editor(
        fair_df, num_rows="fixed", use_container_width=True, key="fair_editor"
    )
    fair = {}
    for role in ROLE_ORDER:
        fair[role] = edited[edited["Ruolo"] == role]["Fair Value"] \
            .astype(int).tolist()

    c1, c2 = st.columns(2)
    if c1.button("💾 Salva configurazione", type="primary"):
        session = asta_core.new_session(budget=int(budget), fair=fair)
        st.session_state["session"] = session
        asta_core.save_session(session)
        st.success("Configurazione salvata")
        st.rerun()
    sessions = asta_core.list_sessions()
    if sessions:
        labels = {str(p): p.stem.replace("config_", "") for p in sessions}
        pick = c2.selectbox("Carica configurazione", [""] + list(labels))
        if pick:
            loaded = asta_core.load_session(pick)
            st.session_state["session"] = loaded
            st.session_state["cfg_budget"] = int(loaded["meta"]["budget"])
            st.session_state["_loaded_fair"] = loaded["meta"].get("fair")
            st.session_state.pop("fair_editor", None)
            st.rerun()

    st.divider()
    st.subheader("🎚️ Pesi classifica")
    st.caption(
        "Punteggio = media pesata normalizzata dei componenti, ognuno in 0–1 "
        "per ruolo (percentile). Peso 0 = componente disattivata. La "
        "robustezza infortuni premia. Starter/Set-pieces contano di più per "
        "P e D che per A (automatico). L'affidabilità d'impiego è un proxy "
        "trasparente, non una stima inventata dei minuti. Al salvataggio la "
        "classifica (rank e cluster) viene ricalcolata."
    )
    saved = load_ranking_weights()
    method = saved.get("_method", DEFAULT_METHOD)
    method_labels = {
        "blend": "Blend modello + pesi (consigliato)",
        "model": "Modello predittivo (FM attesa)",
        "manual": "Pesi manuali",
    }
    chosen = st.selectbox(
        "Metodo classifica",
        list(method_labels),
        format_func=lambda m: method_labels[m],
        index=list(method_labels).index(
            method if method in method_labels else "blend"
        ),
        key="rank_method",
    )
    wcols = st.columns(2)
    weights_edit = {}
    for i, (key, label) in enumerate(WEIGHT_LABELS.items()):
        weights_edit[key] = wcols[i % 2].slider(
            label, 0.0, 2.0, float(saved[key]), 0.1, key=f"w_{key}"
        )
    if st.button("💾 Salva pesi classifica", type="primary"):
        weights_edit["_method"] = chosen
        save_ranking_weights(weights_edit)
        st.cache_data.clear()
        st.success("Pesi salvati — classifica ricalcolata")
        st.rerun()
    with st.expander("📊 Diagnostica modello (Spearman ρ)"):
        rho = backtest_predictor(players_df())
        cols = st.columns(len(ROLE_ORDER))
        for col, role in zip(cols, ROLE_ORDER):
            v = rho.get(role, float("nan"))
            col.metric(role, f"{v:.2f}" if pd.notna(v) else "n/d")
        st.caption(
            "ρ = correlazione di Spearman tra la FM prevista dal modello e "
            "la FM reale dell'ultima stagione, con validazione incrociata a "
            "5 fold. Più alto è meglio; sotto 0.3 il modello è debole e "
            "conviene il metodo manuale. Il blend usa pesi prudenti diversi "
            "per ruolo (P 35%, D 55%, C 60%, A 65% modello); non viene "
            "presentato come un backtest temporale finché non avremo snapshot "
            "storici di presenze/minuti."
        )


def role_count(role):
    return int((players_df()["Ruolo"] == role).sum())


def player_card(player, info):
    cap = info["cap"]
    st.markdown(
        f'<div class="bid-card"><div>'
        f'<div class="bid-label">Massimo da offrire</div>'
        f'<div class="bid-value">{cap} crediti</div>'
        f'<div class="bid-meta">Cluster {info["cluster"]} · fair value '
        f'del ruolo</div></div>'
        f'<div class="bid-stop">STOP oltre {cap}</div></div>',
        unsafe_allow_html=True,
    )
    identity = st.columns(3)
    identity[0].metric("Squadra", player["Squadra"])
    identity[1].metric("Ruolo", player["Ruolo"])
    identity[2].metric("Cluster", f"{info['cluster']}")
    if pd.notna(player["FM"]):
        fm_disp = f"{player['FM']:.2f}"
    elif pd.notna(player["FMEst"]):
        fm_disp = f"~{player['FMEst']:.2f} (stima)"
    else:
        fm_disp = "-"
    pred = player.get("PredFM")
    performance = st.columns(4)
    performance[0].metric("FantaMedia", fm_disp)
    qa = player["QA"]
    performance[1].metric(
        "Quotazione QA", f"{qa:.0f}" if pd.notna(qa) else "-"
    )
    performance[2].metric(
        "FM attesa (modello)", f"{pred:.2f}" if pd.notna(pred) else "-"
    )
    confidence = float(player.get("DataConfidence", 0.0))
    performance[3].metric(
        "Confidenza dati", f"{confidence:.0%}",
        help="Quanto sono complete e stabili le informazioni disponibili: stagioni di FM, fonti e stime.",
    )
    profile = st.columns(3)
    profile[0].metric(
        "Affidabilità impiego", f"{float(player.get('Availability', 0)):.0%}",
        help="Proxy basato su formazione probabile e robustezza agli infortuni; non è una previsione di minuti.",
    )
    profile[1].metric("Valore stagione", f"{float(player.get('SeasonValue', 0)):.0%}")
    profile[2].metric("Upside", f"{float(player.get('Upside', 0)):.0%}")
    if pd.notna(player.get("Minuti")) or pd.notna(player.get("xGI90")):
        advanced = st.columns(5)
        advanced[0].metric("Minuti storici", f"{player['Minuti']:.0f}")
        advanced[1].metric("Titolare storico", f"{player['Titolarita']:.0f}")
        advanced[2].metric("xG/90", f"{player['xG90']:.2f}" if pd.notna(player['xG90']) else "-")
        advanced[3].metric("xA/90", f"{player['xA90']:.2f}" if pd.notna(player['xA90']) else "-")
        advanced[4].metric("xG+xA/90", f"{player['xGI90']:.2f}" if pd.notna(player['xGI90']) else "-")
        if pd.notna(player.get("GareSaltate")):
            st.caption(f"Storico infortuni importato: {player['GareSaltate']:.0f} gare saltate")

    st.markdown(
        f"{player['Nome']} è nel **cluster {info['cluster']}** dei {info['role']} "
        f"(ordinati per punteggio). Fair value del cluster: **{cap}**. "
        f"Se il prezzo sale sopra {cap}, esci dall'asta."
    )
    with st.expander("🧮 Breakdown punteggio"):
        comps = pd.DataFrame({
            "Componente": [
                "FantaMedia (FCP)", "FVM (Gazzetta)", "Algoritmo FCP",
                "Titolare", "Set-pieces", "Attributi", "Robustezza",
                "Affidabilità impiego", "Modello FM attesa",
                "xG + xA per 90",
            ],
            "Contributo": [
                float(player.get("C_FM", 0)), float(player.get("C_FVM", 0)),
                float(player.get("C_ALG", 0)), float(player.get("C_Starter", 0)),
                float(player.get("C_SetPieces", 0)), float(player.get("C_Tags", 0)),
                float(player.get("C_Injury", 0)),
                float(player.get("C_Availability", 0)),
                float(player.get("C_Model", 0)),
                float(player.get("C_ExpectedOutput", 0)),
            ],
        })
        st.dataframe(comps, hide_index=True)
        st.caption(f"Punteggio totale: {player['Score']:.3f} · "
                   f"rank {int(player['Rank'])}/{role_count(player['Ruolo'])}")
    with st.expander("Tabella fair value per il ruolo"):
        table = pd.DataFrame({
            "Cluster": list(range(1, len(info["fair_table"]) + 1)),
            "Fair value": info["fair_table"],
        })
        st.dataframe(table, use_container_width=True, hide_index=True)


def render_players():
    players = players_df()
    if players.empty:
        st.warning("Nessun dato giocatori — scarica le quotazioni nella tab Setup")
        return

    excluded = load_excluded()
    session = st.session_state.get("session")
    bought = asta_core.purchased_names(session) if session else set()
    players = players[~players["Nome"].isin(excluded | bought)].copy()

    clicked = st.session_state.pop("search_click", None)
    if clicked:
        st.session_state["nom_search"] = clicked
        st.session_state["nom_select"] = clicked

    st.header("📊 Ranking e schede giocatori")
    st.markdown(
        '<p class="section-note">Cerca un nome per aprire la scheda completa, '
        'oppure filtra direttamente il ranking per ruolo e cluster.</p>',
        unsafe_allow_html=True,
    )
    if bought:
        st.caption(f"{len(bought)} giocatori già registrati nell’asta live non sono mostrati qui.")
    st.text_input("Cerca giocatore", key="nom_search")
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
        budget, fair = get_config()
        session = {"meta": {"budget": budget, "fair": fair}}
        info = asta_core.coach(session, player)
        player_card(player, info)

    st.divider()
    st.subheader("🎯 Filtra per ruolo e cluster")
    c1, c2, c3 = st.columns([1, 1, 2])
    role = c1.selectbox("Ruolo", ["Tutti"] + ROLE_ORDER, key="filtro_ruolo")
    sub = players if role == "Tutti" else players[players["Ruolo"] == role]
    clusters = sorted(
        int(c) for c in sub["Cluster"].dropna().unique() if pd.notna(c)
    )
    cluster = c2.selectbox(
        "Cluster", ["Tutti"] + clusters, key="filtro_cluster"
    )
    c3.caption("Spunta i giocatori già battuti all'asta e rimuovili: "
               "spariranno da questa tab.")
    view = sub.copy()
    if cluster != "Tutti":
        view = view[view["Cluster"] == cluster]
    view = view.sort_values(["Ruolo", "Rank"], ascending=[True, True])
    summary = st.columns(3)
    summary[0].metric("Giocatori visibili", len(view))
    summary[1].metric(
        "Score migliore", f"{view['Score'].max():.3f}" if not view.empty else "-"
    )
    summary[2].metric("Filtro cluster", str(cluster))
    budget, fair = get_config()
    view = view[["Nome", "Squadra", "Ruolo", "FM", "QA", "Cluster",
                 "Score", "SeasonValue", "DataConfidence"]].copy()
    view["Massimo da offrire"] = view.apply(
        lambda r: fair.get(r["Ruolo"], [])[
            min(int(r["Cluster"]) - 1, len(fair.get(r["Ruolo"], [1])) - 1)
        ],
        axis=1,
    )
    view["Escludi"] = False
    edited = st.data_editor(
        view, use_container_width=True, hide_index=True, key="cluster_editor",
        column_config={
            "Nome": st.column_config.TextColumn(disabled=True),
            "Squadra": st.column_config.TextColumn(disabled=True),
            "Ruolo": st.column_config.TextColumn(disabled=True),
            "FM": st.column_config.NumberColumn(disabled=True),
            "QA": st.column_config.NumberColumn(disabled=True),
            "Cluster": st.column_config.NumberColumn(disabled=True),
            "Score": st.column_config.NumberColumn(
                disabled=True, format="%.3f",
                help="Punteggio composito ponderato (Setup → Pesi classifica)"),
            "SeasonValue": st.column_config.NumberColumn(
                "Valore stagione", disabled=True, format="%.0f%%",
                help="Qualità attesa + affidabilità d'impiego."),
            "DataConfidence": st.column_config.NumberColumn(
                "Confidenza", disabled=True, format="%.0f%%",
                help="Copertura e profondità dei dati disponibili."),
            "Massimo da offrire": st.column_config.NumberColumn(
                disabled=True, help="Massimo da offrire all'asta"),
        },
    )
    st.caption(f"{len(view)} giocatori nel cluster selezionato · "
               f"budget {budget}")
    if cluster != "Tutti":
        for start in range(0, len(view), 5):
            cols = st.columns(5)
            for col, (_, r) in zip(cols, view.iloc[start:start + 5].iterrows()):
                with col:
                    if st.button(
                        f"🔍 {r['Nome']}", key=f"pick_{r['Nome']}",
                        use_container_width=True
                    ):
                        st.session_state["search_click"] = r["Nome"]
                        st.rerun()
        st.caption("👆 Clicca un giocatore per aprirlo in 'Cerca giocatore'")
    if st.button("🗑️ Rimuovi selezionati (già battuti all'asta)",
                 type="primary"):
        to_remove = edited[edited["Escludi"] == True]["Nome"].tolist()
        if to_remove:
            excluded |= set(to_remove)
            save_excluded(excluded)
            st.success(f"Rimossi {len(to_remove)} giocatori")
            st.rerun()
        else:
            st.warning("Nessun giocatore selezionato")

    with st.expander(f"♻️ Ripristina giocatori rimossi ({len(excluded)})"):
        if not excluded:
            st.caption("Nessun giocatore rimosso")
        for name in sorted(excluded):
            c1, c2 = st.columns([3, 1])
            c1.write(name)
            if c2.button("Ripristina", key=f"rest_{name}"):
                excluded.discard(name)
                save_excluded(excluded)
                st.rerun()

    st.divider()
    with st.expander("📋 Massimo per cluster (per ruolo)", expanded=True):
        budget, fair = get_config()
        st.caption(
            f"Budget: {budget} · cluster fissi da {10} giocatori "
            f"(10 partecipanti, uno a testa per cluster)"
        )
        for role in ROLE_ORDER:
            table = fair[role]
            rows = [
                {"Cluster": c, "Massimo da offrire": table[min(c - 1, len(table) - 1)]}
                for c in range(1, len(table) + 1)
            ]
            st.markdown(f"**{role}**")
            st.dataframe(
                pd.DataFrame(rows), use_container_width=True, hide_index=True,
            )


def render_live_auction():
    session = st.session_state.get("session")
    if not session:
        st.warning("Prima salva budget e fair value nella tab Setup: serviranno per l’asta live.")
        return
    asta_core.ensure_session(session)
    meta = session["meta"]
    teams = meta["teams"]
    summaries = asta_core.auction_summary(session)
    own = next(item for item in summaries if item["team"] == meta["my_team"])
    all_players = players_df()
    available = all_players
    unavailable = load_excluded() | asta_core.purchased_names(session)
    available = available[~available["Nome"].isin(unavailable)].copy()
    available_records = available.to_dict("records")

    st.header("🎯 Assistente asta personale")
    st.markdown(
        '<p class="section-note">Cerca il giocatore chiamato: ottieni un prezzo per provarci, '
        'uno stop invalicabile e un piano B prima del prossimo rilancio.</p>',
        unsafe_allow_html=True,
    )
    top = st.columns(4)
    top[0].metric("La mia squadra", own["team"])
    top[1].metric("Crediti rimasti", own["remaining"])
    top[2].metric("Spesi", own["spent"])
    top[3].metric("Giocatori presi", len(own["purchases"]))
    role_progress = " · ".join(
        f"{role} {own['by_role'][role]}/{meta['slots'][role]}" for role in ROLE_ORDER
    )
    st.caption(f"Rosa: {role_progress}")

    freshness = source_freshness()
    freshness_columns = st.columns(len(freshness))
    for column, item in zip(freshness_columns, freshness):
        column.metric(item["label"], f"{item['state']} {item['value']}")
    st.caption("🟢 aggiornato · 🟠 da aggiornare · 🔴 dato non disponibile")

    alerts = roster_alerts(own, all_players, meta["slots"])
    if alerts:
        st.warning("**Attenzioni sulla mia rosa**\n\n" + "\n".join(
            f"- {alert}" for alert in alerts
        ))

    with st.expander("🧭 Piano personale e watchlist", expanded=True):
        st.caption(
            "Scegli come vuoi spendere nei reparti. Questa scelta modifica solo il "
            "prezzo consigliato: lo STOP del cluster non cambia mai."
        )
        cols = st.columns(len(ROLE_ORDER))
        priorities = {}
        for col, role in zip(cols, ROLE_ORDER):
            current_plan = closest_role_plan(meta["role_priorities"].get(role, 1.0))
            plan = col.selectbox(
                f"{role}: strategia", list(ROLE_PLAN_CHOICES),
                index=list(ROLE_PLAN_CHOICES).index(current_plan),
                key=f"role_plan_{role}",
            )
            priorities[role] = ROLE_PLAN_CHOICES[plan]
        st.caption(
            "Risparmia = non inseguire quel reparto · Equilibrio = nessuna preferenza · "
            "Spingi = avvicinati di più al prezzo consigliato se il profilo ti serve."
        )
        if st.button("💾 Salva piano personale"):
            meta["role_priorities"] = priorities
            asta_core.save_session(session)
            st.success("Piano personale salvato")
            st.rerun()
        watchlist = session["watchlist"]
        st.markdown("**La tua lista privata**")
        st.caption(
            "A = obiettivo principale · B = alternativa valida · C = occasione solo a buon prezzo. "
            "La lettera modifica il prezzo consigliato, mai lo STOP del cluster."
        )
        if watchlist:
            watch_df = pd.DataFrame(watchlist).rename(columns={
                "name": "Giocatore", "club": "Squadra", "role": "Ruolo",
                "tier": "Priorità", "note": "Nota",
            })
            st.dataframe(watch_df, hide_index=True, use_container_width=True)
            remove_name = st.selectbox(
                "Rimuovi dalla watchlist", [item["name"] for item in watchlist],
                key="watchlist_remove",
            )
            if st.button("Rimuovi obiettivo"):
                asta_core.remove_watchlist_item(session, remove_name)
                asta_core.save_session(session)
                st.rerun()
        else:
            st.caption("Nessun obiettivo salvato: aggiungine uno dalla scheda del giocatore chiamato.")

    with st.expander("⚙️ Partecipanti", expanded=False):
        st.caption("I nomi servono soltanto a registrare l’asta. Devono restare 10 e distinti.")
        edited_teams = st.data_editor(
            pd.DataFrame({"Partecipante": teams}),
            num_rows="fixed", hide_index=True, use_container_width=True,
            key="team_editor",
        )
        selected_own = st.selectbox(
            "La mia squadra", teams,
            index=teams.index(meta["my_team"]), key="my_team_editor",
        )
        if st.button("💾 Salva partecipanti"):
            updated = [str(name).strip() for name in edited_teams["Partecipante"]]
            if len(updated) != 10 or len(set(updated)) != 10 or not all(updated):
                st.error("Inserisci esattamente 10 nomi distinti e non vuoti.")
            else:
                renamed = dict(zip(teams, updated))
                for purchase in session["purchases"]:
                    purchase["team"] = renamed.get(purchase["team"], purchase["team"])
                meta["teams"] = updated
                meta["my_team"] = renamed.get(selected_own, updated[0])
                asta_core.save_session(session)
                st.success("Partecipanti salvati")
                st.rerun()

    st.divider()
    st.subheader("⚡ Giocatore chiamato ora")
    if available.empty:
        st.info("Non ci sono giocatori disponibili: controlla le esclusioni o annulla un acquisto.")
    else:
        st.text_input("Cerca giocatore chiamato", key="auction_query", placeholder="Es. Lautaro")
        query = st.session_state.get("auction_query", "")
        matches = suggest_player(query, available) if len(query) >= 2 else pd.DataFrame()
        if len(query) < 2:
            st.caption("Scrivi almeno due lettere per cercare il giocatore.")
        elif matches.empty:
            st.warning("Nessun giocatore disponibile con questo nome.")
        else:
            choices = matches["Nome"].tolist()
            picked = st.selectbox(
                "Giocatore", choices,
                format_func=lambda name: (
                    f"{name} · {matches[matches['Nome'] == name].iloc[0]['Squadra']} "
                    f"({matches[matches['Nome'] == name].iloc[0]['Ruolo']})"
                ),
                key="auction_player",
            )
            player = matches[matches["Nome"] == picked].iloc[0]
            if st.session_state.get("called_player") != picked:
                st.session_state["called_player"] = picked
                st.session_state["current_auction_price"] = 1
                saved_watch = next(
                    (item for item in session["watchlist"]
                     if item["name"] == picked),
                    None,
                )
                st.session_state["watch_tier"] = (
                    saved_watch["tier"] if saved_watch else "B"
                )
                st.session_state["watch_note"] = (
                    saved_watch["note"] if saved_watch else ""
                )
            current_price = st.number_input(
                "Prezzo attuale", min_value=1, max_value=max(1, int(meta["budget"])),
                step=1, key="current_auction_price",
            )
            advice = asta_core.auction_advice(
                session, player, available_records, current_price=current_price
            )
            verdict_messages = {
                "PUNTA": st.success,
                "SOLO SE È UNA PRIORITÀ": st.warning,
                "LASCIA": st.error,
                "NON COMPRARE": st.error,
            }
            verdict_messages[advice["verdict"]](
                f"{advice['verdict']} — prezzo consigliato fino a "
                f"{advice['recommended']} crediti."
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Punta fino a", f"{advice['recommended']} crediti")
            c2.metric("Massimo personale", f"{advice['personal_max']} crediti")
            c3.metric("STOP cluster", f"{advice['fixed_cap']} crediti")
            st.caption(
                f"{advice['role_left']} slot {player['Ruolo']} da riempire · "
                f"riserva {advice['reserve']} crediti per gli altri slot · "
                f"{advice['alternatives']} alternative comparabili disponibili."
            )
            st.markdown(
                f"**Perché:** qualità nel cluster {advice['quality']:.0%}, "
                f"necessità reparto {advice['need']:.0%}, strategia "
                f"{closest_role_plan(advice['priority']).lower()}."
            )
            if advice["watch_tier"]:
                st.info(
                    f"Watchlist {advice['watch_tier']}: "
                    f"{WATCHLIST_EXPLANATIONS[advice['watch_tier']]}"
                )

            alternatives = available[
                (available["Ruolo"] == player["Ruolo"]) &
                (available["Nome"] != player["Nome"])
            ].copy()
            alternatives = alternatives[
                (alternatives["Cluster"] <= int(player["Cluster"]) + 1) &
                (alternatives["Rank"] > int(player["Rank"]))
            ]
            if alternatives.empty:
                alternatives = available[
                    (available["Ruolo"] == player["Ruolo"]) &
                    (available["Nome"] != player["Nome"])
                ].copy()
            alternatives = alternatives.sort_values(["Cluster", "Rank"]).head(3)
            comparison_rows = [{
                "Scelta": "Chiamato ora",
                "Giocatore": player["Nome"],
                "Squadra": player["Squadra"],
                "Cluster": int(player["Cluster"]),
                "Valore stagione": f"{float(player.get('SeasonValue', 0)):.0%}",
                "Titolare": "Sì" if float(player.get("Starter", 0)) >= 0.5 else "No / dubbio",
                "Punta fino a": advice["recommended"],
                "STOP": advice["fixed_cap"],
            }]
            for _, alternative in alternatives.iterrows():
                alternative_advice = asta_core.auction_advice(
                    session, alternative, available_records
                )
                comparison_rows.append({
                    "Scelta": "Alternativa",
                    "Giocatore": alternative["Nome"],
                    "Squadra": alternative["Squadra"],
                    "Cluster": int(alternative["Cluster"]),
                    "Valore stagione": f"{float(alternative.get('SeasonValue', 0)):.0%}",
                    "Titolare": (
                        "Sì" if float(alternative.get("Starter", 0)) >= 0.5
                        else "No / dubbio"
                    ),
                    "Punta fino a": alternative_advice["recommended"],
                    "STOP": alternative_advice["fixed_cap"],
                })
            st.markdown("**Confronto diretto — il chiamato contro il tuo piano B:**")
            st.dataframe(
                pd.DataFrame(comparison_rows), hide_index=True,
                use_container_width=True,
            )
            if alternatives.empty:
                st.caption("Non ci sono alternative comparabili ancora disponibili in questo ruolo.")

            w1, w2, w3 = st.columns([1, 1, 2])
            watch_tier = w1.selectbox(
                "Tipo obiettivo", ["A", "B", "C"],
                format_func=lambda tier: f"{tier} — {WATCHLIST_EXPLANATIONS[tier]}",
                key="watch_tier",
            )
            watch_note = w2.text_input("Nota privata", key="watch_note", placeholder="es. titolare")
            if w3.button("⭐ Salva come obiettivo", use_container_width=True):
                asta_core.save_watchlist_item(session, player, watch_tier, watch_note)
                asta_core.save_session(session)
                st.success(f"{player['Nome']} aggiunto alla watchlist {watch_tier}.")
                st.rerun()

            st.markdown("**Registra esito della chiamata**")
            c1, c2 = st.columns(2)
            buyer = c1.selectbox(
                "Aggiudicato a", teams, index=teams.index(meta["my_team"]),
                key="auction_buyer",
            )
            final_price = c2.number_input(
                "Prezzo finale", min_value=1, max_value=max(1, int(meta["budget"])),
                value=int(current_price), step=1, key="auction_price",
            )
            if st.button("✅ Registra acquisto", type="primary"):
                try:
                    purchase = asta_core.record_purchase(session, player, buyer, final_price)
                    asta_core.save_session(session)
                    over_cap = purchase["price"] - purchase["cap"]
                    message = f"Registrato {purchase['name']} a {buyer} per {purchase['price']} crediti."
                    if over_cap > 0:
                        message += f" {over_cap} sopra il cap consigliato."
                    st.success(message)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    st.divider()
    market = asta_core.market_snapshot(session)
    market_text = " · ".join(
        f"{item['role']}: {'+' if item['delta'] >= 0 else ''}{item['delta']:.0%} "
        f"su cap ({item['count']} acquisti)" for item in market if item["count"]
    )
    if market_text:
        st.caption(f"Termometro mercato: {market_text}")
    st.subheader("Situazione partecipanti")
    board = pd.DataFrame([
        {
            "Partecipante": item["team"], "Crediti rimasti": item["remaining"],
            "Spesi": item["spent"], "Rosa": len(item["purchases"]),
            **item["by_role"],
        }
        for item in summaries
    ])
    st.dataframe(board, hide_index=True, use_container_width=True)

    st.subheader("Rosa e annullamento")
    roster_team = st.selectbox("Mostra rosa di", teams, key="roster_team")
    roster = asta_core.team_summary(session, roster_team)["purchases"]
    if not roster:
        st.caption(f"{roster_team} non ha ancora acquisti registrati.")
    else:
        roster_df = pd.DataFrame(roster)[
            ["name", "club", "role", "price", "cap", "cluster", "created"]
        ].rename(columns={
            "name": "Giocatore", "club": "Squadra", "role": "Ruolo",
            "price": "Prezzo", "cap": "Cap", "cluster": "Cluster", "created": "Registrato",
        })
        st.dataframe(roster_df, hide_index=True, use_container_width=True)
        undo_name = st.selectbox(
            "Annulla acquisto", [purchase["name"] for purchase in roster],
            key="undo_purchase",
        )
        if st.button("↩️ Annulla acquisto selezionato"):
            asta_core.undo_purchase(session, undo_name)
            asta_core.save_session(session)
            st.success(f"Acquisto di {undo_name} annullato.")
            st.rerun()


def render_formazioni():
    lineups = load_lineups()
    if lineups.empty:
        st.warning("Nessuna formazione — usa 'Aggiorna tutto per l’asta' nella tab Setup")
        return

    updated_at = datetime.fromtimestamp(scraper.FORMAZIONI.stat().st_mtime)
    st.header("🧩 Probabili formazioni")
    st.markdown(
        f"**Ultimo aggiornamento:** {updated_at:%d/%m/%Y %H:%M}  \n"
        "Fonte: Fantacalcio.it per undici, ballottaggi e indisponibili; "
        "Fantacalciopedia per i tiratori. Per aggiornare usa il pulsante unico "
        "nella tab Setup."
    )

    st.markdown(
        "**Legenda:** "
        '<span style="background:#c62828;color:white;border-radius:6px;'
        'padding:1px 8px;font-size:12px">⚽ Rigorista</span> '
        '<span style="background:#1565c0;color:white;border-radius:6px;'
        'padding:1px 8px;font-size:12px">🚩 Angoli</span> '
        '<span style="background:#b8860b;color:white;border-radius:6px;'
        'padding:1px 8px;font-size:12px">🎯 Punizioni</span> — '
        "⭐ = nome da monitorare per bonus — "
        "* = fantamedia stimata (nessuna stagione disponibile)",
        unsafe_allow_html=True,
    )

    teams = sorted(lineups["Squadra"].unique())
    team_rows = {team: lineups[lineups["Squadra"] == team] for team in teams}
    attention_cols = ["Ballottaggi", "Squalificati", "Infortunati", "InDubbio"]
    attention_teams = [
        team for team, rows in team_rows.items()
        if any(str(rows.iloc[0].get(col, "")).strip() for col in attention_cols)
    ]
    bonus_teams = [
        team for team, rows in team_rows.items()
        if rows[["Rigorista", "Punizioni", "Angoli"]].any(axis=None)
    ]
    overview = st.columns(4)
    overview[0].metric("Squadre", len(teams))
    overview[1].metric("Titolari attesi", len(lineups))
    overview[2].metric("Squadre con bonus", len(bonus_teams))
    overview[3].metric("Da monitorare", len(attention_teams))

    c1, c2, c3 = st.columns([2, 2, 2])
    team_choice = c1.selectbox(
        "Squadra", ["Tutte"] + teams, key="lineup_team_filter"
    )
    focus = c2.selectbox(
        "Mostra", ["Tutte", "Solo bonus", "Solo da monitorare"],
        key="lineup_focus_filter",
    )
    c3.text_input("Trova giocatore", key="lineup_player_filter", placeholder="Es. Barella")
    player_query = st.session_state.get("lineup_player_filter", "").strip().lower()
    shown_teams = teams if team_choice == "Tutte" else [team_choice]
    if focus == "Solo bonus":
        shown_teams = [team for team in shown_teams if team in bonus_teams]
    elif focus == "Solo da monitorare":
        shown_teams = [team for team in shown_teams if team in attention_teams]
    if player_query:
        shown_teams = [
            team for team in shown_teams
            if team_rows[team]["Nome"].str.lower().str.contains(
                player_query, regex=False
            ).any()
        ]
    st.caption(f"{len(shown_teams)} squadre visibili · aggiorna i dati dalla tab Setup il giorno dell’asta")

    def player_row(row):
        flagged = bool(row["Rigorista"] or row["Punizioni"] or row["Angoli"])
        bg = "#e8f5e9" if flagged else "#fafafa"
        border = "1px solid #81c784" if flagged else "1px solid #e0e0e0"
        badges = ""
        if row["Rigorista"]:
            badges += ('<span style="background:#c62828;color:white;'
                       'border-radius:6px;padding:1px 8px;font-size:11px">'
                       '⚽ Rigorista</span> ')
        if row["Angoli"]:
            badges += ('<span style="background:#1565c0;color:white;'
                       'border-radius:6px;padding:1px 8px;font-size:11px">'
                       '🚩 Angoli</span> ')
        if row["Punizioni"]:
            badges += ('<span style="background:#b8860b;color:white;'
                       'border-radius:6px;padding:1px 8px;font-size:11px">'
                       '🎯 Punizioni</span> ')
        ruolo = row["Ruolo"] or "?"
        fm = f"{row['FM']:.2f}" if pd.notna(row["FM"]) else "-"
        if pd.notna(row["FM"]) and row.get("FMImputed"):
            fm += " *"
        cl = ""
        if pd.notna(row.get("Cluster")) and row.get("Cluster") != "":
            cl = (f'<span style="float:right;color:#1b5e20;font-weight:bold;'
                  f'font-size:11px">C{int(row["Cluster"])}</span>')
        return (
            f'<div style="background:{bg};border:{border};padding:7px 9px;'
            f'border-radius:7px;margin:2px 0;font-size:13px">'
            f'{cl}<b>{row["Nome"]}</b> <span style="color:#666;font-size:11px">'
            f'{ruolo} · FM {fm}</span>  {badges}</div>'
        )

    if not shown_teams:
        st.info("Nessuna squadra corrisponde ai filtri selezionati.")
        return
    per_row = 2 if team_choice == "Tutte" else 1
    for start in range(0, len(shown_teams), per_row):
        cols = st.columns(per_row)
        for col, team in zip(cols, shown_teams[start:start + per_row]):
            sub = lineups[lineups["Squadra"] == team]
            modulo = sub.iloc[0]["Modulo"] if len(sub) else "?"
            with col:
                st.markdown(
                    f'<div style="background:#12372a;color:#f4fbf5;border:1px solid #12372a;'
                    f'border-radius:10px;padding:9px 11px;margin:8px 0 6px">'
                    f'<b style="font-size:15px">{team}</b> '
                    f'<span style="color:#d7eadb;font-size:12px">— {modulo} · '
                    f'{len(sub)} titolari attesi</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                for role in ROLE_ORDER + [""]:
                    role_rows = sub[sub["Ruolo"].fillna("") == role]
                    if role_rows.empty:
                        continue
                    label = role if role else "Ruolo da verificare"
                    st.caption(label)
                    for _, row in role_rows.iterrows():
                        st.markdown(player_row(row), unsafe_allow_html=True)
                bench = sub[sub["Panchina"].fillna("") != ""][
                    ["Panchina", "PanchinaRuolo", "PanchinaFM", "PanchinaCluster"]
                ].drop_duplicates()
                if not bench.empty:
                    bench_labels = []
                    for _, substitute in bench.iterrows():
                        fm = (f"FM {substitute['PanchinaFM']:.2f}"
                              if pd.notna(substitute["PanchinaFM"]) else "FM -")
                        cluster = (f"C{int(substitute['PanchinaCluster'])}"
                                   if pd.notna(substitute["PanchinaCluster"])
                                   and substitute["PanchinaCluster"] != "" else "")
                        bench_labels.append(
                            " · ".join(filter(None, [
                                str(substitute["Panchina"]),
                                str(substitute["PanchinaRuolo"]), fm, cluster,
                            ]))
                        )
                    st.caption("Panchina / coperture: " + "  |  ".join(bench_labels))
                notes = []
                labels = {
                    "Ballottaggi": "🔄 Ballottaggi",
                    "Squalificati": "⛔ Squalificati",
                    "Infortunati": "🩹 Infortunati",
                    "InDubbio": "❓ In dubbio",
                    "Diffidati": "🟨 Diffidati",
                }
                for field, label in labels.items():
                    value = str(sub.iloc[0].get(field, "")).strip()
                    if value:
                        notes.append(f"**{label}:** {value}")
                if notes:
                    st.warning("  \n".join(notes))


def main():
    inject_styles()
    st.markdown(
        '<div class="app-hero">'
        '<h1>⚽ Asta Coach</h1>'
        '<p>Ranking per ruolo, cluster da 10 e massimo da offrire: '
        'decidi in fretta senza inseguire il prezzo.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.title("⚽ Asta Coach")
    budget, _ = get_config()
    st.sidebar.metric("Budget attivo", f"{budget} crediti")
    session = st.session_state.get("session")
    if session:
        asta_core.ensure_session(session)
        own = asta_core.team_summary(session, session["meta"]["my_team"])
        st.sidebar.metric("Miei crediti rimasti", own["remaining"])
        st.sidebar.caption(
            f"Configurazione: budget {session['meta']['budget']} · "
            f"salvata il {session['meta']['created'].replace('T', ' ')}"
        )
        if st.sidebar.button("💾 Salva configurazione"):
            asta_core.save_session(session)
            st.sidebar.success("Salvata")
    else:
        st.sidebar.caption("Configurazione predefinita · salva un setup per "
                           "conservarlo")

    tab_setup, tab_auction, tab_form = st.tabs(
        ["Setup", "Asta live · giocatori", "Formazioni"]
    )
    with tab_setup:
        render_setup()
    with tab_auction:
        render_live_auction()
        st.divider()
        render_players()
    with tab_form:
        render_formazioni()


if __name__ == "__main__":
    main()
