import streamlit as st
import pandas as pd
from streamlit_tags import st_tags

# Load data
url = "https://raw.githubusercontent.com/Blazerss/Applicazione-fantacalcio/main/fanta.csv"
df = pd.read_csv(url)
df.set_index("Nome", inplace=True)

st.title("Fanta Dashboard")

# Tags input
keywords = st_tags(
    label='# Inserisci giocatore:',
    text='Press enter to add more',
    value="",
    suggestions=df.index.tolist()
)

statistiche = ["Pv", "Mv", "Fm", "Gf", "Gs", "Rp", "Rc", "R+", "R-", "Ass", "Amm", "Esp", "Au"]

# Helper functions
def format_name(name):
    tokens = name.split()
    parole_da_verificare = ['di', 'de', 'del']
    if len(tokens) > 1 and tokens[0].lower() in parole_da_verificare:
        return f"{tokens[0].capitalize()} {tokens[1].capitalize()}"
    return tokens[0].capitalize() if tokens else "N/A"

def display_player_stats(player_name, season_suffix):
    """Display statistics for a player for a given season."""
    col_data = df.loc[player_name]
    st.write(f"**{player_name} ({season_suffix.replace('_','-')})**")
    stats_cols = [f"{s} {season_suffix}" for s in statistiche]

    if not col_data[stats_cols].isna().all():
        st.write(f"Partite Giocate: {int(col_data[stats_cols][0])}")
        st.write(f"Media Voto: {col_data[stats_cols][1]}")
        st.write(f"Fanta Media: {col_data[stats_cols][2]}")
        role = col_data["R"]
        if role != "P":
            st.write(f"Goal Fatti: {int(col_data[stats_cols][3])}")
            st.write(f"Rigori Calciati: {int(col_data[stats_cols][6])}")
            st.write(f"Rigori Segnati: {int(col_data[stats_cols][7])}")
            st.write(f"Rigori Sbagliati: {int(col_data[stats_cols][8])}")
            st.write(f"Assist: {int(col_data[stats_cols][9])}")
        else:
            st.write(f"Goal Subiti: {int(col_data[stats_cols][4])}")
            st.write(f"Rigori Parati: {int(col_data[stats_cols][5])}")
        st.write(f"Ammunizioni: {int(col_data[stats_cols][10])}")
        st.write(f"Espulsioni: {int(col_data[stats_cols][11])}")
        st.write(f"Autogol: {int(col_data[stats_cols][12])}")
    else:
        st.write("Statistiche non disponibili.")

def display_general_info(player_name):
    """Display general info for a player."""
    col_data = df.loc[player_name]
    st.write(f"Squadra: {col_data['Squadra']}")
    status = "Hyped" if col_data["Diff."] >= 0 else "No hype"
    st.write(f"Status: {status}")
    st.write(f"Attributi: {col_data['attributi']}")
    st.write(f"Resistenza Infortuni: {col_data['Res. Inf.']}")

# Display info in columns
cols = st.columns(5, gap="medium")
seasons = ["24_25", "23_24", "22_23", "21_22"]

for idx, col in enumerate(cols):
    with col:
        if keywords:
            season = seasons[idx] if idx < len(seasons) else None
            st.write(f"**Stagione {season.replace('_','-')}**")
            for k in keywords:
                name = format_name(k)
                display_general_info(name) if idx == 0 else display_player_stats(name, season)

# Build team table
col1, col2 = st.columns(2)
with col1:
    squadra_input = st_tags(label="Squadra", text='Press enter to add more', suggestions=df.index.tolist())
with col2:
    crediti_input = st_tags(label="Crediti", text='Press enter to add more')

rosa = [format_name(p) for p in squadra_input]
df1 = pd.DataFrame(columns=["Rosa", "Squadra", "Costo", "Crediti Rimasti"])
totale_cred = 0

for i, gioc in enumerate(rosa):
    costo = int(crediti_input[i]) if i < len(crediti_input) else 0
    rimasti = 500 - totale_cred - costo
    df1 = pd.concat([df1, pd.DataFrame([{
        "Rosa": gioc,
        "Squadra": df.loc[gioc, "Squadra"],
        "Costo": costo,
        "Crediti Rimasti": rimasti
    }])], ignore_index=True)
    totale_cred += costo

if not df1.empty:
    df1.set_index("Rosa", inplace=True)
    st.table(df1)
