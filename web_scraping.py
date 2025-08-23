from bs4 import BeautifulSoup as bs
import requests
import csv
import time

ruoli = ['Portieri', 'Difensori', 'Centrocampisti', 'Trequartisti', 'Attaccanti']

def safe_text(tag, default="N/A"):
    return tag.get_text().strip() if tag else default

calciatori = []
for ruolo in ruoli:
    url = f"https://www.fantacalciopedia.com/lista-calciatori-serie-a/{ruolo.lower()}/"
    try:
        html = requests.get(url, timeout=10)
        soup = bs(html.content, "html.parser")
        calciatori += [a.get("href") for a in soup.select("div.col_full.giocatore a") if a.get("href")]
    except Exception as e:
        print(f"Failed to load {url}: {e}")
print(calciatori)
print("**********************************************")
print(list(set(calciatori)))
time.sleep(10)

with open('asta1.csv', 'w', newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Nome", "ALG FTP", "FM 23-24", "FM 22-23", "FM 21-22", "attributi", "Res. Inf."])

    for link in calciatori:
        try:
            html2 = requests.get(link, timeout=10)
            soup2 = bs(html2.content, "html.parser")

            # Extract player name
            nome_div = soup2.find("div", class_="fancy-title title-bottom-border")
            nome = safe_text(nome_div.h1 if nome_div and nome_div.h1 else None)
            tokens = nome.split()
            parole_da_verificare = ['di', 'de', 'del']
            if len(tokens) > 1 and tokens[0].lower() in parole_da_verificare:
                nome = f"{tokens[0].capitalize()} {tokens[1].capitalize()}"
            elif tokens:
                nome = tokens[0]
            else:
                nome = "N/A"

            # Extract attributes
            attrs_div = soup2.find("div", class_="col_full center mc_hookEvolution")
            attrs = []
            if attrs_div:
                for div in attrs_div.find_all("div", class_="col_one_fourth"):
                    span = div.find("span", class_="stickdanpic")
                    attrs.append(safe_text(span))

            # Extract fantavoto
            fm_div = soup2.find("div", class_="col_three_fourth")
            FM = []
            if fm_div:
                for div in fm_div.find_all("div", class_="col_one_fourth"):
                    span = div.find("span", class_="stickdan")
                    FM.append(safe_text(span))
            while len(FM) < 4:
                FM.append("N/A")

            # Extract Res. Inf.
            inf_divs = soup2.find_all("div", class_="counter counter-inherit counter-instant")
            inf = safe_text(inf_divs[-1]) if inf_divs else "N/A"

            writer.writerow([nome, FM[0], FM[1], FM[2], FM[3], attrs, inf])


        except Exception as e:
            print(f"Failed to scrape {link}: {e}")
