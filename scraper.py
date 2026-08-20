import argparse
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup as bs

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PLAYERS_FCP = DATA_DIR / "players_fcp.csv"
QUOTAZIONI = DATA_DIR / "quotazioni.csv"
FORMAZIONI = DATA_DIR / "formazioni.csv"
SET_PIECES = DATA_DIR / "set_pieces.csv"
ADVANCED_STATS = DATA_DIR / "advanced_stats.csv"
PANCHINARI = DATA_DIR / "panchinari.csv"
# Manual, auditable corrections for transfers announced after a source's
# seasonal quotation cache was published. Rows are kept separate from scraped
# data so a later refresh cannot silently reintroduce an ineligible player.
ROSTER_OVERRIDES = DATA_DIR / "roster_overrides.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
}


class ScrapeError(Exception):
    pass

FCP_ROLES = [
    ("P", "https://www.fantacalciopedia.com/lista-calciatori-serie-a/portieri/"),
    ("D", "https://www.fantacalciopedia.com/lista-calciatori-serie-a/difensori/"),
    ("C", "https://www.fantacalciopedia.com/lista-calciatori-serie-a/centrocampisti/"),
    ("C", "https://www.fantacalciopedia.com/lista-calciatori-serie-a/trequartisti/"),
    ("A", "https://www.fantacalciopedia.com/lista-calciatori-serie-a/attaccanti/"),
]

QUOTAZIONI_URL = "https://www.fantacalcio.it/quotazioni-fantacalcio"
LINEUPS_URL = "https://www.fantacalcio.it/probabili-formazioni-serie-a"
SET_PIECES_URL = "https://www.fantacalcio.it/rigoristi-serie-a"
# Formazioni-tipo con le alternative ruolo per ruolo (Titolare/Panchinaro).
# L'articolo è stagionale: aggiorna l'URL alla nuova stagione quando esce.
PANCHINARI_URL = ("https://www.sosfanta.com/asta-fantacalcio/"
                  "seriea-tutte-formazioni-tipo-fantacalcio-2026-2027-"
                  "asta-consigli-chi-prendere/")
FCP_SET_PIECES_URL = ("https://www.fantacalciopedia.com/articoli-fcp/"
                       "consigli-fantacalcio/216-rigoristi-e-tiratori-2026-27.html")

TEAM_SLUGS = {
    "atalanta": "Atalanta", "bologna": "Bologna", "cagliari": "Cagliari",
    "como": "Como", "fiorentina": "Fiorentina", "frosinone": "Frosinone",
    "genoa": "Genoa", "inter": "Inter", "juventus": "Juventus",
    "lazio": "Lazio", "lecce": "Lecce", "milan": "Milan", "monza": "Monza",
    "napoli": "Napoli", "parma": "Parma", "roma": "Roma",
    "sassuolo": "Sassuolo", "torino": "Torino", "udinese": "Udinese",
    "venezia": "Venezia",
}


def normalize_name(name):
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().replace("'", "").replace(".", "").replace("-", " ")
    return re.sub(r"\s+", " ", name).strip()


def _get(url, timeout=25):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp
    except ScrapeError:
        raise
    except Exception as exc:
        raise ScrapeError(f"Errore di rete scaricando {url}: {exc}") from exc


def _clean(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


# Colonne accettate dagli export più comuni (FBref/FotMob). Il file salvato
# localmente usa sempre lo schema italiano sottostante, così data_loader resta
# indipendente dal sito che hai usato per esportare i dati.
ADVANCED_STAT_ALIASES = {
    "NomeStats": ["player", "name", "nome", "giocatore"],
    "SquadraStats": ["squad", "team", "squadra", "club"],
    "Presenze": ["mp", "matches played", "matches", "appearances", "presenze"],
    "Titolarita": ["starts", "started", "titolarita", "starts made"],
    "Minuti": ["min", "minutes", "minutes played", "minuti"],
    "xG": ["xg", "expected goals"],
    "xA": ["xa", "expected assists", "xag"],
    "xGI90": ["xg + xa per 90", "xg+xa per 90", "xg+xA/90", "xgi90"],
    "Gol": ["gls", "goals", "gol"],
    "Assist": ["ast", "assists", "assist"],
    "Gialli": ["crdy", "yellow cards", "gialli"],
    "Rossi": ["crdr", "red cards", "rossi"],
    "GiorniInfortunio": ["days", "days injured", "injury days", "giorni infortunio"],
    "GareSaltate": ["games missed", "matches missed", "gare saltate"],
    "GolSubiti": ["ga", "goals against", "gol subiti", "reti subite", "gol subiti subiti"],
    "GolSubiti90": ["ga90", "ga per 90", "goals against per 90",
                    "gol subiti per 90", "gol subiti/90", "reti subite per 90"],
    "CleanSheet": ["cs", "clean sheets", "cleansheets", "clean sheet",
                   "porte inviolate"],
}


def _stat_column(frame, aliases):
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(col).lower()): col
        for col in frame.columns
    }
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]", "", alias.lower())
        if key in normalized:
            return normalized[key]
    return None


def import_advanced_stats(source, progress_cb=None):
    """Normalize a FBref/FotMob CSV export and save it for offline ranking."""
    try:
        raw = pd.read_csv(source)
    except Exception as exc:
        raise ScrapeError(f"CSV statistiche non leggibile: {exc}") from exc
    if raw.empty:
        raise ScrapeError("Il CSV statistiche è vuoto")

    out = pd.DataFrame(index=raw.index)
    for target, aliases in ADVANCED_STAT_ALIASES.items():
        col = _stat_column(raw, aliases)
        if col is None:
            out[target] = "" if target in {"NomeStats", "SquadraStats"} else float("nan")
        else:
            out[target] = raw[col]
    if out["NomeStats"].replace("", float("nan")).isna().all():
        raise ScrapeError("CSV senza una colonna giocatore (Player, Name, Nome o Giocatore)")

    out["NomeStats"] = out["NomeStats"].fillna("").astype(str).str.strip()
    out["SquadraStats"] = out["SquadraStats"].fillna("").astype(str).str.strip()
    for col in out.columns.difference(["NomeStats", "SquadraStats"]):
        out[col] = pd.to_numeric(
            out[col].astype(str).str.replace(",", ".", regex=False)
            .str.replace(r"[^0-9.\-]", "", regex=True),
            errors="coerce",
        )
    out = out[out["NomeStats"] != ""].drop_duplicates(
        ["NomeStats", "SquadraStats"], keep="first"
    )
    DATA_DIR.mkdir(exist_ok=True)
    out.to_csv(ADVANCED_STATS, index=False)
    if progress_cb:
        coverage = int(out["Minuti"].notna().sum())
        progress_cb(f"{len(out)} giocatori importati ({coverage} con minuti)")
    return out


def scrape_fantacalciopedia(limit=None, delay=0.4, progress_cb=None):
    DATA_DIR.mkdir(exist_ok=True)
    seen = set()
    links = []
    for role, url in FCP_ROLES:
        html = _get(url)
        soup = bs(html.content, "html.parser")
        for a in soup.select("div.col_full.giocatore a"):
            href = a.get("href")
            if href and href not in seen:
                seen.add(href)
                links.append((role, href))
        if progress_cb:
            progress_cb(f"{role} list loaded: {len(seen)} unique players")

    if limit:
        links = links[:limit]

    rows = []
    for i, (role, link) in enumerate(links):
        try:
            html = _get(link)
            soup = bs(html.content, "html.parser")

            ndiv = soup.find("div", class_="fancy-title title-bottom-border")
            name = _clean(ndiv.h1.get_text()) if ndiv and ndiv.h1 else ""

            attrs_div = soup.find("div", class_="col_full center mc_hookEvolution")
            attrs = []
            if attrs_div:
                for div in attrs_div.find_all("div", class_="col_one_fourth"):
                    span = div.find("span", class_="stickdanpic")
                    attrs.append(_clean(span.get_text()) if span else "N/A")

            fm_div = soup.find("div", class_="col_three_fourth")
            fm = []
            if fm_div:
                for div in fm_div.find_all("div", class_="col_one_fourth"):
                    span = div.find("span", class_="stickdan")
                    fm.append(_clean(span.get_text()) if span else "N/A")
            while len(fm) < 4:
                fm.append("N/A")

            counters = soup.find_all("div", class_="counter counter-inherit counter-instant")
            res_inf = _clean(counters[-1].get_text()) if counters else "N/A"

            team = ""
            for a in soup.find_all("a", href=True):
                txt = _clean(a.get_text())
                if "torna a rosa" in txt.lower():
                    team = re.sub(r"(?i)^\s*(>>>\s*)?torna a rosa\s*", "", txt)
                    break

            rows.append(
                {
                    "NomeFCP": name,
                    "Ruolo": role,
                    "SquadraFCP": team,
                    "ALG": fm[0] if fm else "N/A",
                    "FM1": fm[1] if len(fm) > 1 else "N/A",
                    "FM2": fm[2] if len(fm) > 2 else "N/A",
                    "FM3": fm[3] if len(fm) > 3 else "N/A",
                    "Attributi": " | ".join(attrs),
                    "ResInf": res_inf,
                }
            )
        except Exception as exc:
            if progress_cb:
                progress_cb(f"failed {link}: {exc}")
        if progress_cb:
            progress_cb(f"{i + 1}/{len(links)} players scraped")
        time.sleep(delay)

    df = pd.DataFrame(rows)
    df.to_csv(PLAYERS_FCP, index=False)
    return df


def scrape_quotazioni(progress_cb=None):
    DATA_DIR.mkdir(exist_ok=True)
    html = _get(QUOTAZIONI_URL)
    soup = bs(html.content, "html.parser")

    rows = []
    for tr in soup.select("tr.player-row"):
        name_el = tr.select_one("th.player-name span")
        team_el = tr.select_one("td.player-team")
        qi = tr.select_one("td.player-classic-initial-price")
        qa = tr.select_one("td.player-classic-current-price")
        fvm = tr.select_one("td.player-classic-fvm")
        m_qi = tr.select_one("td.player-mantra-initial-price")
        m_qa = tr.select_one("td.player-mantra-current-price")
        m_fvm = tr.select_one("td.player-mantra-fvm")
        link = tr.select_one("th.player-name a")
        slug = ""
        if link and link.get("href"):
            m = re.search(r"/squadre/([a-z-]+)/", link["href"])
            slug = m.group(1) if m else ""

        rows.append(
            {
                "NomeGaz": _clean(name_el.get_text()) if name_el else "",
                "SquadraGaz": slug.upper(),
                "SquadraNome": TEAM_SLUGS.get(slug, ""),
                "Ruolo": tr.get("data-filter-role-classic", "").upper(),
                "QI": _clean(qi.get_text()) if qi else "",
                "QA": _clean(qa.get_text()) if qa else "",
                "FVM": _clean(fvm.get_text()) if fvm else "",
                "M_QI": _clean(m_qi.get_text()) if m_qi else "",
                "M_QA": _clean(m_qa.get_text()) if m_qa else "",
                "M_FVM": _clean(m_fvm.get_text()) if m_fvm else "",
            }
        )

    df = pd.DataFrame(rows)
    for col in ["QI", "QA", "FVM", "M_QI", "M_QA", "M_FVM"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["NomeGaz"] != ""]
    df.to_csv(QUOTAZIONI, index=False)
    if progress_cb:
        progress_cb(f"{len(df)} players with quotes")
    return df


def _formation_display(code):
    if not code:
        return "?"
    return "-".join(list(code))


def _card_players(card, selector):
    names, roles = [], []
    for li in card.select(selector):
        name_el = li.select_one("a.player-name span")
        if not name_el:
            continue
        names.append(_clean(name_el.get_text()))
        role_el = li.select_one("span.role")
        roles.append(role_el.get("data-value", "") if role_el else "")
    return names, roles


def _team_detail(match, section_class, team_index):
    """Read one team's availability notes from the current lineup page."""
    section = match.select_one(f"section.{section_class}")
    if not section:
        return ""
    contents = section.select(":scope > div.content")
    if team_index >= len(contents):
        return ""
    content = contents[team_index]
    if content.select_one(".empty-list-message"):
        return ""
    items = []
    for li in content.select("li"):
        name = _clean(li.select_one(".player-name").get_text()) \
            if li.select_one(".player-name") else ""
        detail = _clean(li.select_one(".description").get_text()) \
            if li.select_one(".description") else ""
        label = ": ".join(value for value in [name, detail] if value)
        if label:
            items.append(label)
    if items:
        return " | ".join(items)
    return _clean(content.get_text(" ", strip=True))


def _team_availability(match, team_index):
    return {
        "Ballottaggi": _team_detail(match, "ballots", team_index),
        "Squalificati": _team_detail(match, "suspendeds", team_index),
        "Diffidati": _team_detail(match, "cautioneds", team_index),
        "Infortunati": _team_detail(match, "injureds", team_index),
        "InDubbio": _team_detail(match, "dubts", team_index),
    }


def scrape_lineups(progress_cb=None):
    DATA_DIR.mkdir(exist_ok=True)
    html = _get(LINEUPS_URL)
    soup = bs(html.content, "html.parser")

    rows = []
    for match in soup.select("li.match-item"):
        # Layout corrente: un campo con due div.team e i dettagli per squadra.
        team_nodes = match.select("div.pitch > div.team")
        team_names = match.select(".match-pill .team-name")
        if len(team_nodes) == 2 and len(team_names) >= 2:
            for team_index, team_node in enumerate(team_nodes):
                starters = [
                    _clean(player.get_text())
                    for player in team_node.select(
                        "ul.team-lineup li.player a.player-name"
                    )
                ]
                rows.append({
                    "Squadra": _clean(team_names[team_index].get_text()),
                    "Modulo": team_node.get("data-team-formation", "?"),
                    "Titolari": "|".join(starters),
                    "RuoliTitolari": "",
                    "Panchina": "",
                    "RuoliPanchina": "",
                    **_team_availability(match, team_index),
                })
            continue
        for card in match.select("div.card.team-card"):
            team_el = card.select_one("header h3.team-name")
            if not team_el:
                continue
            team = _clean(team_el.get_text())
            modulo_el = card.select_one("header .team-formation")
            modulo = _clean(modulo_el.get_text()) if modulo_el else "?"
            starters, s_roles = _card_players(
                card, "ul.player-list.starters li"
            )
            bench, b_roles = _card_players(
                card, "ul.player-list.reserves li"
            )
            rows.append(
                {
                    "Squadra": team,
                    "Modulo": modulo,
                    "Titolari": "|".join(starters),
                    "RuoliTitolari": "|".join(s_roles),
                    "Panchina": "|".join(bench),
                    "RuoliPanchina": "|".join(b_roles),
                    **_team_availability(match, len(rows) % 2),
                }
            )

    if not rows:
        for match in soup.select("li.match-item"):
            h3s = match.select("div.row.col-sm h3.team-name")
            uls = match.select("div.pitch ul.team-lineup")
            for h3, ul in zip(h3s, uls):
                team = _clean(h3.get_text())
                modulo = _formation_display(ul.get("data-formation"))
                players = [
                    _clean(li.select_one("a.player-name span").get_text())
                    for li in ul.select("li.player")
                    if li.select_one("a.player-name span")
                ]
                rows.append(
                    {"Squadra": team, "Modulo": modulo,
                     "Titolari": "|".join(players),
                     "RuoliTitolari": "", "Panchina": "",
                     "RuoliPanchina": "", **_team_availability(match, len(rows) % 2)}
                )

    df = pd.DataFrame(rows)
    df.to_csv(FORMAZIONI, index=False)
    if progress_cb:
        progress_cb(f"{len(df)} formazioni (20 squadre attese)")
    return df


def _split_names(raw):
    raw = re.sub(r"\.{2,}", " ", raw)
    parts = re.split(r"[,\n;]+", raw)
    names = []
    for p in parts:
        p = p.strip()
        p = re.sub(r"\s+", " ", p)
        p = re.sub(r"\s*-\s*", " ", p)
        if p:
            names.append(p)
    return names


def scrape_set_pieces(progress_cb=None):
    DATA_DIR.mkdir(exist_ok=True)
    html = _get(SET_PIECES_URL)
    soup = bs(html.content, "html.parser")
    rows = []
    # Fonte primaria: stessa redazione delle quotazioni e delle probabili
    # formazioni. Espone una gerarchia esplicita, non una lista piatta.
    for card in soup.select("div.card.team-card"):
        team_el = card.select_one("header.team-info .team-name")
        if not team_el:
            continue
        team = _clean(team_el.get_text())
        for col in card.select("div.row.row-responsive > div.col"):
            heading = _clean(col.select_one("header").get_text() if col.select_one("header") else "")
            tipo = {"Rigori": "Rigorista", "Calci piazzati": "Piazzati"}.get(heading)
            if not tipo:
                continue
            for order, item in enumerate(col.select("ol.pill-list li"), start=1):
                name_el = item.select_one("a.player-name")
                name = _clean(name_el.get_text()) if name_el else ""
                if name:
                    rows.append({"Squadra": team, "Giocatore": name,
                                 "Tipo": tipo, "Ordine": order})

    if not rows:
        # Fallback storico: mantiene l'app utilizzabile se la pagina primaria
        # cambia struttura o non è momentaneamente disponibile.
        html = _get(FCP_SET_PIECES_URL)
        soup = bs(html.content, "html.parser")
        text = soup.get_text("\n")

        start = text.find("Tiratori Atalanta")
        if start < 0:
            start = text.find("Rigoristi e Tiratori Serie A")
        body = text[start:] if start >= 0 else text
        for marker in ["Autore", "Copyrights", "Guida Fantacalcio 2022"]:
            k = body.find(marker)
            if k > 0:
                body = body[:k]
                break

        team_pattern = re.compile(
            r"Tiratori\s+([A-Z][A-Za-zÀ-ÿ ']+?)\s+20\d\d/27"
        )
        type_pattern = re.compile(
            r"(Rigoristi in ordine|Calci di punizione|Calci d'angolo):"
        )
        tipo_map = {
            "Rigoristi in ordine": "Rigorista",
            "Calci di punizione": "Punizioni",
            "Calci d'angolo": "Angoli",
        }
        junk_prefixes = ("da aggiornare", "autore", "bio", "ultimo aggiornamento")

        matches = list(team_pattern.finditer(body))
        for idx, m in enumerate(matches):
            team = m.group(1).strip()
            block_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
            block = body[m.end():block_end]
            types = list(type_pattern.finditer(block))
            for j, tm in enumerate(types):
                sec_end = types[j + 1].start() if j + 1 < len(types) else len(block)
                section = block[tm.end():sec_end]
                for order, name in enumerate(_split_names(section), start=1):
                    name = name.rstrip(".")
                    if not name or len(name) > 40 \
                            or name.lower().startswith(junk_prefixes) \
                            or any(ch in name for ch in "0123456789:@/()|©"):
                        continue
                    rows.append({"Squadra": team, "Giocatore": name,
                                 "Tipo": tipo_map[tm.group(1)], "Ordine": order})

    df = pd.DataFrame(rows).drop_duplicates()
    if df.empty:
        raise ScrapeError("Nessun rigorista o specialista trovato nelle fonti disponibili")
    df.to_csv(SET_PIECES, index=False)
    if progress_cb:
        progress_cb(f"{len(df)} indicazioni set-pieces "
                    f"({df['Squadra'].nunique()} squadre)")
    return df


def _parse_formation_tipo(line):
    """Parse a sosfanta 'Formazione-tipo' line into role-by-role slots.

    Positions are separated by ';' (P, D, C, A), players by ',', and the
    alternatives for the same spot by '/'. Returns rows with the slot order
    preserved so the first name is the favourite starter.
    """
    roles = ["P", "D", "C", "A"]
    text = _clean(line).rstrip(".").strip()
    rows = []
    for index, group in enumerate(text.split(";")):
        if index >= len(roles):
            break
        role = roles[index]
        for slot_no, slot in enumerate(group.split(","), start=1):
            names = [n.strip() for n in slot.split("/") if n.strip()]
            if names:
                rows.append({"Ruolo": role, "Slot": slot_no, "Nomi": "|".join(names)})
    return rows


STATISTICHE_URL = "https://www.fantacalcio.it/statistiche-serie-a"
# Codici squadra 3 lettere usati nelle statistiche → nomi canonici.
STAT_TEAM_CODES = {
    "ATA": "Atalanta", "BOL": "Bologna", "CAG": "Cagliari", "CAR": "Carpi",
    "CHI": "Chievo", "CRO": "Crotone", "COM": "Como", "CRE": "Cremonese",
    "EMP": "Empoli", "FIO": "Fiorentina", "FRO": "Frosinone", "GEN": "Genoa",
    "INT": "Inter", "JUV": "Juventus", "LAZ": "Lazio", "LEC": "Lecce",
    "MIL": "Milan", "MON": "Monza", "NAP": "Napoli", "PAL": "Palermo",
    "PAR": "Parma", "PES": "Pescara", "PIS": "Pisa", "ROM": "Roma",
    "SAL": "Salernitana", "SAM": "Sampdoria", "SAS": "Sassuolo",
    "SPA": "SPAL", "SPE": "Spezia", "TOR": "Torino", "UDI": "Udinese",
    "VEN": "Venezia", "VER": "Verona", "BEN": "Benevento",
}
# Stagioni completate scaricate di default per gol subiti/rigori parati.
DEFAULT_STATISTICHE_SEASONS = ["2025-26", "2024-25", "2023-24"]


def _num(text):
    try:
        return float(_clean(text).replace(",", "."))
    except (ValueError, AttributeError):
        return float("nan")


def _rig_parts(text):
    """Parse a 'scored / taken' penalty cell like '4 / 5'."""
    parts = str(_clean(text)).split("/")
    if len(parts) != 2:
        return float("nan"), float("nan")
    return _num(parts[0]), _num(parts[1])


def scrape_statistiche_season(season, progress_cb=None):
    """Season stats table (fantacalcio.it) for one Serie A season.

    Includes, for every player: presenze, media voto, fantamedia, gol,
    gol subiti, rigori (segnati/tirati), rigori parati, assist and cards.
    Goalkeepers are the rows with GolSubiti > 0.
    """
    DATA_DIR.mkdir(exist_ok=True)
    html = _get(f"{STATISTICHE_URL}/{season}")
    soup = bs(html.content, "html.parser")
    table = soup.select_one("table#stats")
    if not table:
        raise ScrapeError(
            f"Tabella statistiche non trovata per la stagione {season}: "
            "struttura del sito cambiata?"
        )
    rows = []
    for tr in table.select("tbody tr"):
        name_el = tr.select_one("th.player-name span")
        if not name_el:
            continue

        def cell(key):
            el = tr.select_one(f'td[data-col-key="{key}"]')
            return _clean(el.get_text()) if el else ""

        rig_seg, rig_tir = _rig_parts(cell("rig"))
        rows.append({
            "Nome": _clean(name_el.get_text()),
            "Squadra": cell("sq"),
            "Presenze": _num(cell("pg")),
            "MediaVoto": _num(cell("mv")),
            "Fantamedia": _num(cell("mfv")),
            "Gol": _num(cell("gol")),
            "GolSubiti": _num(cell("gs")),
            "RigoriSegnati": rig_seg,
            "RigoriTirati": rig_tir,
            "RigoriParati": _num(cell("rp")),
            "Assist": _num(cell("ass")),
            "Ammonizioni": _num(cell("amm")),
            "Espulsioni": _num(cell("esp")),
        })
    if not rows:
        raise ScrapeError(f"Nessuna riga estratta per la stagione {season}")
    out = pd.DataFrame(rows)
    path = DATA_DIR / f"statistiche_{season}.csv"
    out.to_csv(path, index=False)
    if progress_cb:
        progress_cb(f"{season}: {len(out)} calciatori salvati in {path.name}")
    return out


def scrape_statistiche(seasons=None, progress_cb=None):
    """Scrape the season stats tables for the requested seasons."""
    seasons = list(seasons or DEFAULT_STATISTICHE_SEASONS)
    frames = []
    for season in seasons:
        frames.append(scrape_statistiche_season(season, progress_cb))
    return pd.concat(frames, ignore_index=True)


def scrape_panchinari(progress_cb=None):
    """Fetch the sosfanta season article and save possible bench players.

    Every slot stores the favourite starter plus his alternatives; a player
    not in the starting XI (or the other way around) is the 'possibile
    panchinaro' shown per player in the Formazioni tab.
    """
    DATA_DIR.mkdir(exist_ok=True)
    html = _get(PANCHINARI_URL)
    soup = bs(html.content, "html.parser")
    rows = []
    for em in soup.find_all("em"):
        label = (em.get_text("", strip=True) or "").lower()
        if "formazione-tipo" not in label:
            continue
        team = None
        for strong in em.find_all_previous("strong"):
            key = normalize_name(strong.get_text("", strip=True))
            if key in TEAM_SLUGS:
                team = TEAM_SLUGS[key]
                break
        if not team:
            continue
        text = em.parent.get_text(" ", strip=True)
        text = re.sub(r"^Formazione-tipo\s*:", "", text, flags=re.IGNORECASE).strip()
        for entry in _parse_formation_tipo(text):
            rows.append({"Squadra": team, **entry})
    if not rows:
        raise ScrapeError(
            "Nessuna formazione-tipo trovata su sosfanta: struttura o URL cambiati?"
        )
    out = pd.DataFrame(rows)
    out.to_csv(PANCHINARI, index=False)
    if progress_cb:
        progress_cb(f"{len(out)} slot panchinari "
                    f"({out['Squadra'].nunique()} squadre)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", action="store_true")
    parser.add_argument("--quotes", action="store_true")
    parser.add_argument("--lineups", action="store_true")
    parser.add_argument("--setpieces", action="store_true")
    parser.add_argument("--panchinari", action="store_true")
    parser.add_argument(
        "--statistiche", nargs="?", const=True, default=False,
        metavar="STAGIONI",
        help="scarica statistiche stagionali (es. 2025-26,2024-25); "
             "senza valore usa le ultime 3 stagioni",
    )
    parser.add_argument("--advanced", metavar="CSV",
                        help="importa CSV storico FBref/FotMob")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.all:
        args.players = True
        args.quotes = True
        args.lineups = True
        args.setpieces = True
        args.panchinari = True
        args.statistiche = True

    if args.players:
        print("scraping fantacalciopedia players...")
        df = scrape_fantacalciopedia(limit=args.limit, progress_cb=print)
        print(f"saved {len(df)} players to {PLAYERS_FCP}")
    if args.quotes:
        print("scraping gazzetta quotes...")
        df = scrape_quotazioni(progress_cb=print)
        print(f"saved {len(df)} players to {QUOTAZIONI}")
    if args.lineups:
        print("scraping expected lineups...")
        df = scrape_lineups(progress_cb=print)
        print(f"saved {len(df)} formazioni to {FORMAZIONI}")
    if args.setpieces:
        print("scraping set pieces...")
        df = scrape_set_pieces(progress_cb=print)
        print(f"saved {len(df)} to {SET_PIECES}")
    if args.panchinari:
        print("scraping probable bench players from sosfanta...")
        df = scrape_panchinari(progress_cb=print)
        print(f"saved {len(df)} bench slots to {PANCHINARI}")
    if args.statistiche:
        seasons = (
            [s.strip() for s in args.statistiche.split(",") if s.strip()]
            if isinstance(args.statistiche, str) else None
        )
        print(f"scraping season stats ({seasons or 'default 3 seasons'})...")
        df = scrape_statistiche(seasons, progress_cb=print)
        print(f"saved {len(df)} season rows to data/statistiche_*.csv")
    if args.advanced:
        df = import_advanced_stats(args.advanced, progress_cb=print)
        print(f"saved {len(df)} players to {ADVANCED_STATS}")
