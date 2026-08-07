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
}

st.set_page_config(page_title="Asta Coach", page_icon="⚽", layout="wide")


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
def load_players():
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
        return session["meta"]["budget"], session["meta"]["fair"]
    return 500, fair_values_scaled(500)


def render_setup():
    st.header("⚙️ Setup")
    st.subheader("🔄 Dati")
    c1, c2, c3 = st.columns(3)
    if c1.button("Scarica quotazioni Gazzetta (veloce)"):
        run_scrape([
            ("Scaricando quotazioni...",
             lambda: scraper.scrape_quotazioni(progress_cb=None)),
        ])
        st.rerun()
    if c2.button("Scarica lista giocatori FCP (lento, ~10 min)"):
        run_scrape([
            ("Scaricando giocatori da fantacalciopedia...",
             lambda: scraper.scrape_fantacalciopedia(progress_cb=None)),
        ])
        st.rerun()
    if c3.button("Scarica formazioni + tiratori (~30s)"):
        run_scrape([
            ("Scaricando formazioni e set-pieces...",
             lambda: scraper.scrape_lineups(progress_cb=None)),
            ("Scaricando tiratori...",
             lambda: scraper.scrape_set_pieces(progress_cb=None)),
        ])
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
            )

    st.divider()
    st.subheader("⚙️ Configurazione cap (fair value per cluster)")
    st.caption(
        "Ogni ruolo è diviso in cluster da 10 giocatori (per fantamedia, "
        "uno per squadra). Il cap di un giocatore è il fair value del suo "
        "cluster: oltre quel prezzo conviene lasciar perdere."
    )
    budget = st.number_input("Budget iniziale", min_value=10, max_value=5000,
                             value=500, step=10, key="cfg_budget")
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
            st.session_state["session"] = asta_core.load_session(pick)
            st.rerun()

    st.divider()
    st.subheader("🎚️ Pesi classifica")
    st.caption(
        "Punteggio = media pesata normalizzata dei componenti, ognuno in 0–1 "
        "per ruolo (percentile). Peso 0 = componente disattivata. La "
        "robustezza infortuni premia. Starter/Set-pieces contano di più per "
        "P e D che per A (automatico). Al salvataggio la classifica (rank e "
        "cluster) viene ricalcolata."
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
            "conviene il metodo manuale."
        )


def role_count(role):
    return int((players_df()["Ruolo"] == role).sum())


def player_card(player, info):
    cap = info["cap"]
    st.markdown(
        f'<div style="background:#1f5c2e;color:white;padding:16px 20px;'
        f'border-radius:10px;font-size:26px;font-weight:bold">'
        f'💰 Massimo da offrire: {cap} crediti</div>',
        unsafe_allow_html=True,
    )
    c = st.columns(6)
    c[0].metric("Squadra", player["Squadra"])
    c[1].metric("Ruolo", player["Ruolo"])
    if pd.notna(player["FM"]):
        fm_disp = f"{player['FM']:.2f}"
    elif pd.notna(player["FMEst"]):
        fm_disp = f"~{player['FMEst']:.2f} (stima)"
    else:
        fm_disp = "-"
    c[2].metric("FantaMedia", fm_disp)
    qa = player["QA"]
    c[3].metric("Quotazione QA", f"{qa:.0f}" if pd.notna(qa) else "-")
    c[4].metric("Cluster", f"{info['cluster']}")
    pred = player.get("PredFM")
    c[5].metric("FM attesa (modello)", f"{pred:.2f}" if pd.notna(pred) else "-")

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
                "Modello FM attesa",
            ],
            "Contributo": [
                float(player.get("C_FM", 0)), float(player.get("C_FVM", 0)),
                float(player.get("C_ALG", 0)), float(player.get("C_Starter", 0)),
                float(player.get("C_SetPieces", 0)), float(player.get("C_Tags", 0)),
                float(player.get("C_Injury", 0)), float(player.get("C_Model", 0)),
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
    players = players[~players["Nome"].isin(excluded)].copy()

    clicked = st.session_state.pop("search_click", None)
    if clicked:
        st.session_state["nom_search"] = clicked
        st.session_state["nom_select"] = clicked

    st.header("🔍 Cerca giocatore")
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
    budget, fair = get_config()
    view = view[["Nome", "Squadra", "Ruolo", "FM", "QA", "Cluster",
                 "Score"]].copy()
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


def render_formazioni():
    lineups = load_lineups()
    if lineups.empty:
        st.warning("Nessuna formazione — premi 'Scarica formazioni + tiratori' "
                   "nella tab Setup")
        return

    st.header("📋 Probabili formazioni — tutte le 20 squadre")
    c1, c2 = st.columns([2, 3])
    if c1.button("🔄 Ricomputa formazioni"):
        run_scrape([
            ("Ricalcolo formazioni...",
             lambda: scraper.scrape_lineups(progress_cb=None)),
            ("Aggiornando tiratori...",
             lambda: scraper.scrape_set_pieces(progress_cb=None)),
        ])
        st.rerun()
    c2.caption("Fonte: fantacalcio.it (11 attesi + panchina) + "
               "fantacalciopedia (tiratori)")

    st.markdown(
        "**Legenda:** "
        '<span style="background:#c62828;color:white;border-radius:6px;'
        'padding:1px 8px;font-size:12px">⚽ Rigorista</span> '
        '<span style="background:#1565c0;color:white;border-radius:6px;'
        'padding:1px 8px;font-size:12px">🚩 Angoli</span> '
        '<span style="background:#b8860b;color:white;border-radius:6px;'
        'padding:1px 8px;font-size:12px">🎯 Punizioni</span> — '
        "sfondo verde = giocatore con probabili bonus — "
        "* = fantamedia stimata (nessuna stagione disponibile)",
        unsafe_allow_html=True,
    )

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
                  f'font-size:11px">Cluster {int(row["Cluster"])}</span>')
        sub = ""
        if row.get("Panchina"):
            pfm = (f"{row['PanchinaFM']:.2f}"
                   if pd.notna(row.get("PanchinaFM")) else "-")
            pcl = ""
            if pd.notna(row.get("PanchinaCluster")) \
                    and row.get("PanchinaCluster") != "":
                pcl = f' · Cluster {int(row["PanchinaCluster"])}'
            sub = (f'<div style="color:#37474f;font-size:11px;'
                   f'margin-top:3px;border-top:1px dashed #ccc;'
                   f'padding-top:3px">↪ Sostituto probabile: '
                   f'<b>{row["Panchina"]}</b> '
                   f'<span style="color:#888">· FM {pfm}{pcl}</span></div>')
        return (
            f'<div style="background:{bg};border:{border};padding:5px 9px;'
            f'border-radius:7px;margin:2px 0;font-size:13px">'
            f'{cl}<b>{row["Nome"]}</b> <span style="color:#666;font-size:11px">'
            f'{ruolo} · FM {fm}</span>  {badges}{sub}</div>'
        )

    teams = sorted(lineups["Squadra"].unique())
    per_row = 4
    for start in range(0, len(teams), per_row):
        cols = st.columns(per_row)
        for col, team in zip(cols, teams[start:start + per_row]):
            sub = lineups[lineups["Squadra"] == team]
            modulo = sub.iloc[0]["Modulo"] if len(sub) else "?"
            with col:
                st.markdown(
                    f'<div style="background:#f0f4f8;border:1px solid #d5dde5;'
                    f'border-radius:10px;padding:8px 10px;margin-bottom:8px">'
                    f'<b style="font-size:14px">{team}</b> '
                    f'<span style="color:#444;font-size:12px">— {modulo}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                for _, row in sub.iterrows():
                    st.markdown(player_row(row), unsafe_allow_html=True)


def main():
    st.sidebar.title("⚽ Asta Coach")
    session = st.session_state.get("session")
    if session:
        st.sidebar.caption(
            f"Configurazione: budget {session['meta']['budget']} · "
            f"salvata il {session['meta']['created'].replace('T', ' ')}"
        )
        if st.sidebar.button("💾 Salva configurazione"):
            asta_core.save_session(session)
            st.sidebar.success("Salvata")

    tab_setup, tab_players, tab_form = st.tabs(["Setup", "Giocatori", "Formazioni"])
    with tab_setup:
        render_setup()
    with tab_players:
        render_players()
    with tab_form:
        render_formazioni()


if __name__ == "__main__":
    main()
