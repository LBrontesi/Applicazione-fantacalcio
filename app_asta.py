import pandas as pd
import streamlit as st

import asta_core
import data_loader
import scraper
from data_loader import (
    ROLE_ORDER, build_lineups, build_players, fair_values_scaled, suggest_player,
)

st.set_page_config(page_title="Asta Coach", page_icon="⚽", layout="wide")


@st.cache_data(show_spinner=False)
def load_players():
    return build_players()


@st.cache_data(show_spinner=False)
def load_lineups():
    players = build_players()
    return build_lineups(players)


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
    if c3.button("Scarica formazioni + tiratori (~30s)"):
        with st.status("Scaricando formazioni e set-pieces...") as status:
            scraper.scrape_lineups(
                progress_cb=lambda m: status.update(label=m))
            scraper.scrape_set_pieces(
                progress_cb=lambda m: status.update(label=m))
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
        fair_df, num_rows="fixed", width="stretch", key="fair_editor"
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


def player_card(player, info):
    cap = info["cap"]
    st.markdown(
        f'<div style="background:#1f5c2e;color:white;padding:16px 20px;'
        f'border-radius:10px;font-size:26px;font-weight:bold">'
        f'💰 Massimo da offrire: {cap} crediti</div>',
        unsafe_allow_html=True,
    )
    c = st.columns(5)
    c[0].metric("Squadra", player["Squadra"])
    c[1].metric("Ruolo", player["Ruolo"])
    c[2].metric("FantaMedia", f"{player['FM']:.2f}" if pd.notna(player["FM"])
                else "-")
    qa = player["QA"]
    c[3].metric("Quotazione QA", f"{qa:.0f}" if pd.notna(qa) else "-")
    c[4].metric("Cluster", f"{info['cluster']}")

    st.markdown(
        f"{player['Nome']} è nel **cluster {info['cluster']}** dei {info['role']} "
        f"(ordinati per fantamedia). Fair value del cluster: **{cap}**. "
        f"Se il prezzo sale sopra {cap}, esci dall'asta."
    )
    with st.expander("Tabella fair value per il ruolo"):
        table = pd.DataFrame({
            "Cluster": list(range(1, len(info["fair_table"]) + 1)),
            "Fair value": info["fair_table"],
        })
        st.dataframe(table, width="stretch", hide_index=True)


def render_players():
    players = players_df()
    if players.empty:
        st.warning("Nessun dato giocatori — scarica le quotazioni nella tab Setup")
        return

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
    with st.expander("📋 Massimo per cluster (per ruolo)", expanded=True):
        budget, fair = get_config()
        st.caption(f"Budget: {budget} · cluster da 10 giocatori")
        for role in ROLE_ORDER:
            table = fair[role]
            rows = [
                {"Cluster": c, "Giocatori": f"dal {1 + (c - 1) * 10}° "
                                           f"al {c * 10}°",
                 "Massimo da offrire": table[min(c - 1, len(table) - 1)]}
                for c in range(1, len(table) + 1)
            ]
            st.markdown(f"**{role}**")
            st.dataframe(
                pd.DataFrame(rows), width="stretch", hide_index=True,
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
        with st.status("Ricalcolo...") as status:
            scraper.scrape_lineups(
                progress_cb=lambda m: status.update(label=m))
            scraper.scrape_set_pieces(
                progress_cb=lambda m: status.update(label=m))
            st.cache_data.clear()
            status.update(label="Fatto!", state="complete")
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
        "sfondo verde = giocatore con probabili bonus",
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

    tab_setup, tab_players, tab_form = st.tabs(
        ["Setup", "Giocatori", "Formazioni"], key="main_tabs"
    )
    with tab_setup:
        render_setup()
    with tab_players:
        render_players()
    with tab_form:
        render_formazioni()


if __name__ == "__main__":
    main()
