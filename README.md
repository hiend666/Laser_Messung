# Messdaten-Auswertung – Laservibrometer

Webbasierte Auswertesoftware für CSV-Messdaten aus Laservibrometern. Die App läuft im Browser und ermöglicht die interaktive Analyse von Weg-, Geschwindigkeits- und Beschleunigungsverläufen ohne Installation zusätzlicher Software.

## Funktionsumfang

### Datenimport
- Bis zu 4 Messkanäle pro Datei
- Vier Dateiformate:
  - **CSV plain** – kommagetrennt, ohne Zeitachse (Samplerate wird manuell angegeben)
  - **Hubmessung** – TAB-getrennt, .txt, Zeitachse in Sekunden
  - **Oszilloskop CSV** – kommagetrennt, Zeitachse in Sekunden
  - **CSX** – Oszilloskop CSV mit eingebetteten JSON-Einstellungen (Kanalnamen, Einheiten, Skalierung)
- Konfigurierbare Abtastrate (Eingabe in Hz oder µs/ms)
- Einstellbare Anzahl zu überspringender Kopfzeilen und maximale Sampleanzahl
- Kanalskaliierung (Oszi-Skale, z. B. 100 mV/V für Kraftsensoren)

### Signalaufbereitung
- **SG-Vorfilter** (Savitzky-Golay) pro Kanal: Glättung der Rohsignale vor Offset-Anwendung
- **Y-Offset** pro Kanal: manuell per Slider oder automatisch (Mittelwert im sichtbaren Bereich)
- **X-Offset** pro Kanal: zeitliche Verschiebung einzelner Kanäle relativ zueinander
- **Crop**: Ansicht auf den Cursor-Bereich zuschneiden (+15 % Rand)

### Diagramm
- Interaktives Mehrkanal-Diagramm mit automatischem Multi-Achsen-Layout
- Kanäle gleicher Einheit teilen eine Achse; bei starker Bereichsabweichung automatisch getrennte Achsen
- Manuelle Y-Grenzen pro Kanal oder über den Y-Slider
- **Gleiche Nulllinie**: alle Kanäle einer Einheit werden auf gemeinsamen Y-Bereich normiert
- Geschwindigkeit (1. Ableitung) und Beschleunigung (2. Ableitung) via Savitzky-Golay-Filter
- Einstellbare Fensterbreite für die Glättung (separat für v und a)
- **Integral**: Fläche unter der Kurve zwischen XA–XB + kumulativer Verlauf
- **Multi-Diagramm**: zeitliche Überlagerung mehrerer Perioden
- Zwei Cursor-Marker (XA, XB) zur Bereichsauswahl per Slider
- Automatische Zeiteinheit-Skalierung (s / ms / µs / ns) je nach sichtbarem Bereich

### Kenngrößen-Analyse

| Zeile | Kenngrößen |
|---|---|
| Zeit & Weg | Δt (A–B), Frequenz, Δs Cursor, Hub |
| Geschwindigkeit | v-mid, Δv Cursor, v-max (Peak), SOP |
| Beschleunigung | a-max Falling, a-min Rising |

### Best-Fit-Rechteck
- Automatische Erkennung eines Rechteck-Signals im Wegsignal
- Berechnung von Hub (Amplitude) und Frequenz
- Einzeichnen des erkannten Rechtecks im Diagramm

### Speed on Point (SOP)
- Bestimmt die Geschwindigkeit an einem definierten Punkt auf der steigenden Flanke des Rechteck-Signals
- Einstellbarer Schwellwert (0–100 % des Hubs, Standard 80 %)
- SOP-Marker wird im Diagramm eingezeichnet

### Datei-Merger (separate Seite)
- Kanäle aus bis zu 4 Dateien zu einer gemeinsamen Datei zusammenführen
- Unterstützte Eingabeformate: **CSV plain**, **Hubmessung**, **Oszilloskop CSV**, **CSX**
  - Bei CSX werden Kanalnamen und Einheiten automatisch aus dem JSON-Block übernommen
- Exportformate: CSV plain, TXT (Hubmessung), Oszilloskop CSV
- Optionale Sampleraten-Angleichung per linearer Interpolation
- Längenanpassung: kürzen oder mit Füllwert auffüllen

### Export
- **PDF** (Querformat A4): Diagramm + Kenngrößen-Tabelle mit Dateiname und Zeitstempel
- **PNG**: Diagramm als Bilddatei
- **CSV / CSX**: Messdaten mit optionalen eingebetteten Einstellungen (Kanalnamen, Einheiten, Skalierung)

## Starten

### Empfohlen: Hilfsskript `run.sh`

```bash
./run.sh
```

Das Skript aktiviert die virtuelle Umgebung und startet den Server mit den nötigen Flags automatisch.

### Manuell (venv aktivieren, dann starten)

```bash
source .venv/bin/activate
streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
```

### Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

Im Dev-Container startet die App automatisch auf Port 8501.

## Technische Voraussetzungen

- Python 3.11
- Virtuelle Umgebung unter `.venv/` (empfohlen)
- Pakete: `streamlit`, `pandas`, `plotly`, `scipy`, `kaleido==0.2.1`, `reportlab`

## Version

`v1.06.00`
