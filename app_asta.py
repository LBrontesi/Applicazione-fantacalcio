import pandas as pd
import streamlit as st

import asta_core
import data_loader
import scraper
from data_loader import ROLE_ORDER, build_players, fair_values_scaled, suggest_player

st.set_page_config(page_title="Asta Coach", page_icon="⚽", layout="wide")


@st.cache_data(show_spinner=False)
def load_players():
    return build_players()


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

    tab_setup, tab_players = st.tabs(
        ["Setup", "Giocatori"], key="main_tabs"
    )
    with tab_setup:
        render_setup()
    with tab_players:
        render_players()


if __name__ == "__main__":
    main()
