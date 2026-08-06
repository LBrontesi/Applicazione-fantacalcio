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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", action="store_true")
    parser.add_argument("--quotes", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.all:
        args.players = True
        args.quotes = True

    if args.players:
        print("scraping fantacalciopedia players...")
        df = scrape_fantacalciopedia(limit=args.limit, progress_cb=print)
        print(f"saved {len(df)} players to {PLAYERS_FCP}")
    if args.quotes:
        print("scraping gazzetta quotes...")
        df = scrape_quotazioni(progress_cb=print)
        print(f"saved {len(df)} players to {QUOTAZIONI}")
