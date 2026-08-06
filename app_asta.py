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
                 f"comprare")
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
        st.subheader("Cerca giocatore e cap")
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
            info = asta_core.coach(session, player)
            player_card(player, info)

            st.markdown("**Registra aggiudicazione**")
            f1, f2, f3 = st.columns(3)
            buyers = list(session["teams"].keys())
            buyer = f1.selectbox("Aggiudicatario", buyers, key="bid_buyer")
            price = f2.number_input(
                "Prezzo (crediti)", min_value=1,
                max_value=session["meta"]["budget"], value=1, step=1,
                key="nom_price",
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
                    note=note,
                )
                asta_core.save_session(session)
                st.session_state.pop("nom_search", None)
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
        st.subheader("📜 Log vendite")
        if session["events"]:
            log = pd.DataFrame(session["events"])
            st.dataframe(
                log[["ts", "player", "buyer", "price", "cap", "note"]],
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
