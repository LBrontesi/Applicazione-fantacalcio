import streamlit as st
import pandas as pd
from matplotlib.testing.compare import crop_to_same
from pandas.core.computation.parsing import tokenize_string
from streamlit_tags import st_tags

# Load data
url = "https://raw.githubusercontent.com/Blazerss/Applicazione-fantacalcio/main/fanta.csv"
df = pd.read_csv(url)
df.set_index("Nome", inplace=True)

st.title("Fanta Dashboard")


# Helper functions
def format_name(name):
    tokens = name.split()
    print(tokens)

    if "'" in tokens[0]:
        tokens = tokens[0].split("'",1)
        print('ciao')
        print(tokens)
        return f"{tokens[0].capitalize()}'{tokens[1].capitalize()}"
    if len(tokens) > 1 :
        return f"{tokens[0].capitalize()} {tokens[1].capitalize()}"
    return tokens[0].capitalize() if tokens else "N/A"


def display_general_info(player_name):
    """Display general info for a player."""
    col_data = df.loc[player_name]
    st.write(f"**Squadra:** {col_data['Squadra']}")
    status = "Hyped" if col_data["Diff."] >= 0 else "No hype"
    st.write(f"**Status:** {status}")
    st.write(f"**Attributi:** {col_data['attributi']}")
    st.write(f"**Resistenza Infortuni:** {col_data['Res. Inf.']}")


def display_season_stats(player_name, season_suffix):
    """Display player statistics for a given season."""
    col_data = df.loc[player_name]
    stats_cols = [f"{s} {season_suffix}" for s in statistiche]

    if col_data[stats_cols].isna().all():
        st.write("Statistiche non disponibili")
        return

    st.write(f"**Stagione {season_suffix.replace('_', '-')}**")
    st.write(f"Partite Giocate: {int(col_data[stats_cols][0])}")
    st.write(f"Media Voto: {col_data[stats_cols][1]}")
    st.write(f"Fanta Media: {col_data[stats_cols][2]}")

    role = col_data["R"]
    if role != "P":  # Non-portiere
        st.write(f"Goal Fatti: {int(col_data[stats_cols][3])}")
        st.write(f"Rigori Calciati: {int(col_data[stats_cols][6])}")
        st.write(f"Rigori Segnati: {int(col_data[stats_cols][7])}")
        st.write(f"Rigori Sbagliati: {int(col_data[stats_cols][8])}")
        st.write(f"Assist: {int(col_data[stats_cols][9])}")
    else:  # Portiere
        st.write(f"Goal Subiti: {int(col_data[stats_cols][4])}")
        st.write(f"Rigori Parati: {int(col_data[stats_cols][5])}")

    st.write(f"Ammunizioni: {int(col_data[stats_cols][10])}")
    st.write(f"Espulsioni: {int(col_data[stats_cols][11])}")
    st.write(f"Autogol: {int(col_data[stats_cols][12])}")


# Input tags
keywords = st_tags(
    label='# Inserisci giocatore:',
    text='Press enter to add more',
    value="",
    suggestions=df.index.tolist()
)

statistiche = ["Pv", "Mv", "Fm", "Gf", "Gs", "Rp", "Rc", "R+", "R-", "Ass", "Amm", "Esp", "Au"]
seasons = ["24_25", "23_24", "22_23", "21_22"]

# Display player info
for k in keywords:
    name = format_name(k)
    with st.expander(name, expanded=True):
        display_general_info(name)
        for season in seasons:
            display_season_stats(name, season)

# Team builder
st.header("Costruisci la tua squadra")
col1, col2 = st.columns(2)
squadra_input=[]
with col1:
    squadra_input = st_tags(label="Seleziona giocatori", text='Press enter to add more', suggestions=df.index.tolist())
    #squadra_input = st.text_input(label="Seleziona giocatori")


with col2:
    crediti_input = st_tags(label="Crediti spesi", text='Press enter to add more')



# Build team DataFrame
rosa = [format_name(p) for p in squadra_input]
totale_cred = 0
team_data = []
for i, gioc in enumerate(rosa):

    costo = int(crediti_input[i]) if i < len(crediti_input) else 0
    rimasti = 500 - totale_cred - costo
    team_data.append({
        "Rosa": gioc,
        "Squadra": df.loc[gioc, "Squadra"],
        "Costo": costo,
        "Crediti Rimasti": rimasti
    })
    totale_cred += costo

if team_data:
    df_team = pd.DataFrame(team_data).set_index("Rosa")
    st.table(df_team)
