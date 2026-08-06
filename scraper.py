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

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
}

FCP_ROLES = [
    ("P", "https://www.fantacalciopedia.com/lista-calciatori-serie-a/portieri/"),
    ("D", "https://www.fantacalciopedia.com/lista-calciatori-serie-a/difensori/"),
    ("C", "https://www.fantacalciopedia.com/lista-calciatori-serie-a/centrocampisti/"),
    ("C", "https://www.fantacalciopedia.com/lista-calciatori-serie-a/trequartisti/"),
    ("A", "https://www.fantacalciopedia.com/lista-calciatori-serie-a/attaccanti/"),
]

QUOTAZIONI_URL = "https://www.fantacalcio.it/quotazioni-fantacalcio"
LINEUPS_URL = "https://www.fantacalcio.it/probabili-formazioni-serie-a"
SET_PIECES_URL = ("https://www.fantacalciopedia.com/articoli-fcp/"
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
    return requests.get(url, headers=HEADERS, timeout=timeout)


def _clean(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def scrape_fantacalciopedia(limit=None, delay=0.4, progress_cb=None):
    DATA_DIR.mkdir(exist_ok=True)
    seen = set()
    links = []
    for role, url in FCP_ROLES:
        html = _get(url)
        html.raise_for_status()
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
    html.raise_for_status()
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


def scrape_lineups(progress_cb=None):
    DATA_DIR.mkdir(exist_ok=True)
    html = _get(LINEUPS_URL)
    html.raise_for_status()
    soup = bs(html.content, "html.parser")

    rows = []
    for match in soup.select("li.match-item"):
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
                     "RuoliPanchina": ""}
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
    html.raise_for_status()
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
    rows = []
    for idx, m in enumerate(matches):
        team = m.group(1).strip()
        block_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        block = body[m.end():block_end]
        types = list(type_pattern.finditer(block))
        for j, tm in enumerate(types):
            sec_end = types[j + 1].start() if j + 1 < len(types) else len(block)
            section = block[tm.end():sec_end]
            for name in _split_names(section):
                name = name.rstrip(".")
                if not name or len(name) > 40 \
                        or name.lower().startswith(junk_prefixes) \
                        or any(ch in name for ch in "0123456789:@/()|©"):
                    continue
                rows.append({"Squadra": team, "Giocatore": name,
                             "Tipo": tipo_map[tm.group(1)]})

    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(SET_PIECES, index=False)
    if progress_cb:
        progress_cb(f"{len(df)} indicazioni set-pieces "
                    f"({df['Squadra'].nunique()} squadre)")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", action="store_true")
    parser.add_argument("--quotes", action="store_true")
    parser.add_argument("--lineups", action="store_true")
    parser.add_argument("--setpieces", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.all:
        args.players = True
        args.quotes = True
        args.lineups = True
        args.setpieces = True

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
