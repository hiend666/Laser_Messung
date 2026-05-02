# Messdaten-Auswertung – Laservibrometer

Webbasierte Auswertesoftware für CSV-Messdaten aus Laservibrometern. Die App läuft im Browser und ermöglicht die interaktive Analyse von Weg-, Geschwindigkeits- und Beschleunigungsverläufen ohne Installation zusätzlicher Software.

## Funktionsumfang

### Datenimport
- CSV-Dateien mit zwei Messkanälen (z. B. Festo-Sensor und DST-Sensor)
- Unterstützt zwei CSV-Formate: numerische Rohdaten (ohne Kopfzeile) und Dateien mit Spaltenbezeichnungen
- Konfigurierbare Abtastrate (Eingabe in Hz oder µs)
- Einstellbare Anzahl zu überspringender Kopfzeilen und maximale Sampleanzahl

### Diagramm
- Interaktives Mehrkanal-Diagramm mit drei Y-Achsen:
  - **Weg** (mm) – linke Achse
  - **Geschwindigkeit** (mm/s) – rechte Achse, ±3200 mm/s
  - **Beschleunigung** (m/s²) – rechte Achse, ±12000 m/s²
- Geschwindigkeit und Beschleunigung werden per Savitzky-Golay-Filter aus dem Wegsignal abgeleitet
- Einstellbare Fensterbreite für die Glättung (separat für v und a)
- Zwei rote Cursor-Marker (XA, XB) zur Bereichsauswahl per Slider
- Crop-Funktion: Ansicht auf den Cursor-Bereich zuschneiden (+15 % Rand)

### Kenngrößen-Analyse
Die berechneten Kenngrößen werden in drei Zeilen angezeigt:

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
- SOP-Marker (×) wird im Diagramm in der Farbe der Geschwindigkeitskurve eingezeichnet

### Beschleunigungspeaks
- **a-max Falling**: größte negative Beschleunigung auf der fallenden Flanke (Kreuz-Marker)
- **a-min Rising**: größte positive Beschleunigung auf der steigenden Flanke (Kreis-Marker)

### Y-Offset-Korrektur
- Manuelle Offset-Eingabe für beide Messkanäle per Slider
- Auto-0-Schaltfläche setzt den Offset automatisch auf den Mittelwert des sichtbaren Bereichs

### Export
- **PDF** (Querformat A4): Diagramm + Kenngrößen-Tabelle mit Dateiname und Zeitstempel
- **PNG**: Diagramm als Bilddatei

## Starten

```bash
# Abhängigkeiten installieren
pip install -r requirements.txt

# App starten
streamlit run app.py
```

Im Dev-Container startet die App automatisch auf Port 8501.

## Technische Voraussetzungen

- Python 3.11
- Pakete: `streamlit`, `pandas`, `plotly`, `scipy`, `kaleido==0.2.1`, `reportlab`

## Version

`v1.00.05`
