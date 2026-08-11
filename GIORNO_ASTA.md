# Guida rapida — giorno dell'asta

## La sera prima

1. Avvia l'app e premi **🚀 Aggiorna tutto per l'asta (~10 min)** nella tab Setup.
2. Se lo usi, importa il CSV delle statistiche avanzate (FBref/FotMob).
3. Controlla che siano visibili giocatori, ranking, formazioni e tiratori.
4. Copia la cartella `data/` in una posizione sicura. In particolare conserva:
   - `data/sessions/`
   - `data/formazioni.csv`
   - `data/players_fcp.csv`
   - `data/quotazioni.csv`
   - `data/set_pieces.csv`
   - `data/advanced_stats.csv` (se importato)

## Prima dell'asta

1. Collega il Mac all'alimentazione.
2. Apri un secondo Terminale e impedisci lo stop del Mac:

```bash
caffeinate -dimsu
```

Lascia questo Terminale aperto fino alla fine dell'asta. Per fermarlo, premi `Ctrl+C`.

3. Avvia l'app nel primo Terminale:

```bash
cd /Users/bro/Desktop/Applicazione-fantacalcio
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 web_app.py
```

4. Apri <http://127.0.0.1:7860/> e verifica che sia tutto visibile.
5. Se vuoi dati più recenti, premi **Aggiorna tutto per l'asta** una sola volta, idealmente 1–2 ore prima: il download dei giocatori FCP può richiedere alcuni minuti.

## Durante l'asta

- Non aggiornare le fonti durante l'asta: ranking, cap, formazioni e statistiche lavorano localmente dopo l'aggiornamento iniziale.
- Usa la tab **Asta live** e registra ogni acquisto subito dopo l'aggiudicazione.
- Tieni un secondo tab o un foglio di emergenza con budget residuo, acquisti e obiettivi principali.
- Non chiudere né il Terminale dell'app né quello con `caffeinate`.

## Se l'app si blocca o il browser si chiude

1. Riapri <http://127.0.0.1:7860/>.
2. Se non risponde, esegui di nuovo il comando di avvio nel primo Terminale.
3. Carica la configurazione/sessione: gli acquisti sono salvati localmente in `data/sessions/`.

La connessione Internet serve solo per aggiornare le fonti. Dopo l'aggiornamento, l'assistente asta resta utilizzabile anche senza rete.
