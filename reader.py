"""
Datei-Einleser und Datenaufbereitung für Messdaten-Auswertung.
Unterstützte Formate: CSV plain, Hubmessung TXT, Oszilloskop CSV.
Kein Streamlit – nur Python/NumPy/Pandas/SciPy.
"""
from __future__ import annotations
import io
import json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# CSX – CSV mit eingebetteten JSON-Einstellungen
# ---------------------------------------------------------------------------
_CSX_MARKER_BEGIN = b"###CSX_SETTINGS_BEGIN###"
_CSX_MARKER_END   = b"###CSX_SETTINGS_END###"


def split_csx(file_bytes: bytes) -> tuple[bytes, dict | None]:
    idx = file_bytes.find(_CSX_MARKER_BEGIN)
    if idx == -1:
        return file_bytes, None
    raw = file_bytes[:idx].rstrip(b"\n").rstrip(b"\r\n")
    json_start = idx + len(_CSX_MARKER_BEGIN)
    json_end   = file_bytes.find(_CSX_MARKER_END, json_start)
    json_bytes = file_bytes[json_start:json_end].strip()
    try:
        return raw, json.loads(json_bytes)
    except Exception:
        return raw, None


# Polynomgrad für alle Savitzky-Golay-Filter
SAVGOL_POLYNOM = 3

# Iterationen des k-Means-Schwellwert-Finders im Rechteck-Fit
_RECHTECK_K_MEANS_ITER = 5


# ===========================================================================
# ZEITACHSE
# ===========================================================================

def build_time_axis(n_samples: int, sr_hz: float, hz_faktor: float = 1000.0) -> np.ndarray:
    """Zeitvektor in der gewählten Einheit für n_samples bei Abtastrate sr_hz (Hz)."""
    return np.arange(n_samples) * (hz_faktor / sr_hz)


def wahl_zeiteinheit(max_s: float) -> tuple[float, str]:
    """Wählt die beste Zeiteinheit für die Anzeige.

    Ziel: Maximalwert soll zwischen 0.02 und 800 liegen.
    Gibt (hz_faktor, einheit_label) zurück:
    - hz_faktor:     Sekunden × hz_faktor = Anzeigewert
    - einheit_label: 's', 'ms', 'µs' oder 'ns'
    """
    kandidaten = [(1.0, 's'), (1e3, 'ms'), (1e6, 'µs'), (1e9, 'ns')]
    for hz_faktor, label in kandidaten:
        val = max_s * hz_faktor
        if 0.020 <= val <= 800:
            return hz_faktor, label
    # Kein perfektes Fenster: erste Einheit nehmen bei der max >= 0.02
    for hz_faktor, label in kandidaten:
        if max_s * hz_faktor >= 0.020:
            return hz_faktor, label
    return 1e9, 'ns'


# ===========================================================================
# SAVITZKY-GOLAY ABLEITUNGEN
# ===========================================================================

def clamp_savgol_fenster(fenster: int, n: int) -> int:
    """Klemmt SG-Fenstergröße auf gültigen Wert (ungerade, < n). Mindestgröße ≥ 5 liegt beim Aufrufer."""
    if fenster >= n:
        fenster = n if n % 2 == 1 else n - 1
    if fenster % 2 == 0:
        fenster -= 1
    return fenster


def glaette_signal(signal: np.ndarray, window_length: int, polyorder: int = 3) -> np.ndarray:
    """SG-Glättung (deriv=0) zur Vorfilterung quantisierter Signale."""
    if len(signal) < 4:
        return signal.copy()
    win = clamp_savgol_fenster(window_length, len(signal))
    poly = min(polyorder, win - 1)
    return savgol_filter(signal, win, poly, deriv=0, mode='mirror')


def berechne_sg_ableitung(
    signal: np.ndarray,
    dt_s: float,
    fenster: int,
    ordnung: int,
    polynom: int = SAVGOL_POLYNOM,
) -> np.ndarray | None:
    """Savitzky-Golay-Ableitung beliebiger Ordnung.

    ordnung 1 → Geschwindigkeit (signal_einheit / s)
    ordnung 2 → Beschleunigung  (signal_einheit / s²)
    Gibt None zurück wenn das Signal zu kurz ist.
    """
    fenster = clamp_savgol_fenster(fenster, len(signal))
    if fenster < 5:
        return None
    return savgol_filter(signal, fenster, polynom, deriv=ordnung, delta=dt_s, mode='mirror')


# ===========================================================================
# OFFSET-ANWENDUNG
# ===========================================================================

def apply_offsets(
    kanal_namen: tuple[str, ...],
    kanal_arrays: tuple,          # tuple von np.ndarray, je Kanal ein Array
    offsets: tuple[float, ...],
    zeit: np.ndarray,
) -> pd.DataFrame:
    """Wendet Y-Offsets an und gibt den verarbeiteten DataFrame zurück."""
    data: dict = {'Zeit (ms)': zeit}
    for name, arr, off in zip(kanal_namen, kanal_arrays, offsets):
        data[name] = np.asarray(arr, dtype=np.float64) + float(off)
    return pd.DataFrame(data)


# ===========================================================================
# FORMATE – ROHEINLESER
# ===========================================================================

def peek_oszilloskop_header(file_bytes: bytes) -> tuple[list[int], list[str]]:
    """Liest Kanal-Indices und Einheiten aus dem Oszilloskop-CSV-Header.

    Gibt (kanal_indices, einheiten) zurück:
    - kanal_indices: z.B. [1, 2, 3] (aktive Kanal-Nummern aus Zeile 1)
    - einheiten:     z.B. ['Ampere', 'Volt', 'Volt'] (ohne Zeiteinheit)
    """
    content = file_bytes.decode('utf-8', errors='ignore')
    lines = content.splitlines()
    if len(lines) < 2:
        return [], []

    header_parts = lines[0].split(',')
    kanal_indices: list[int] = []
    for part in header_parts[1:]:
        part = part.strip()
        if part:
            try:
                kanal_indices.append(int(part))
            except ValueError:
                pass

    unit_parts = lines[1].split(',')
    einheiten = [p.strip() for p in unit_parts[1:] if p.strip()]

    return kanal_indices, einheiten


def peek_csv_plain_einheiten(file_bytes: bytes, n_kanäle: int,
                             erlaubte_einheiten: list[str]) -> list[str | None]:
    """Liest Einheiten aus der ZWEITEN Zeile einer CSV plain Datei.

    Gibt bis zu n_kanäle Einheitenstrings zurück; Einträge sind None wenn die
    Zelle leer ist oder nicht in erlaubte_einheiten enthalten ist.
    """
    try:
        content = file_bytes.decode('utf-8', errors='ignore')
        lines   = [l for l in content.splitlines() if l.strip()]
        if len(lines) < 2:
            return [None] * n_kanäle
        second_line = lines[1]
        sep         = ';' if ';' in second_line else ','
        parts       = [p.strip().strip('"').strip("'") for p in second_line.split(sep)]
        result: list[str | None] = []
        for part in parts[:n_kanäle]:
            result.append(part if part in erlaubte_einheiten else None)
        while len(result) < n_kanäle:
            result.append(None)
        return result
    except Exception:
        return [None] * n_kanäle


def peek_csv_plain_kanalnames(file_bytes: bytes, n_kanäle: int) -> list[str]:
    """Liest Spaltennamen aus der ERSTEN Zeile einer CSV plain Datei.

    Gibt bis zu n_kanäle Namen zurück.
    Leere Liste wenn die erste nicht-leere Zeile numerische Werte enthält.
    """
    try:
        content = file_bytes.decode('utf-8', errors='ignore')
        lines   = [l for l in content.splitlines() if l.strip()]
        if not lines:
            return []
        first_line = lines[0]
        sep        = ';' if ';' in first_line else ','
        parts      = [p.strip().strip('"').strip("'") for p in first_line.split(sep)]
        try:
            float(parts[0].replace(',', '.'))
            return []   # Erste Zelle numerisch → kein Header
        except ValueError:
            pass
        return [p for p in parts if p][:n_kanäle]
    except Exception:
        return []


def _sniff_csv_params(file_bytes: bytes, skip_rows: int) -> tuple[str, str]:
    """Erkennt Trennzeichen und Dezimalzeichen einer CSV-Datei automatisch.

    Gibt (sep, decimal) zurück:
    - (';', ',')  bei deutschem Format  (Semikolon-getrennt, Komma-Dezimal)
    - (';', '.')  bei Semikolon-getrennt mit Punkt-Dezimal
    - (',', '.')  bei englischem Format (Komma-getrennt, Punkt-Dezimal)
    """
    try:
        content    = file_bytes.decode('utf-8', errors='ignore')
        data_lines = [l for l in content.splitlines()[skip_rows:skip_rows + 10] if l.strip()]
        sample     = '\n'.join(data_lines)

        if ';' not in sample:
            return ',', '.'

        # Semikolon ist Trennzeichen – Dezimalzeichen aus erstem Datenwert ableiten
        for line in data_lines:
            if ';' not in line:
                continue
            first_token = line.split(';')[0].strip()
            if ',' in first_token:
                return ';', ','
            if '.' in first_token:
                return ';', '.'
        return ';', ','   # Standard-Annahme: deutsches Format

    except Exception:
        return ',', '.'


def read_csv_plain(
    file_bytes: bytes,
    skip_rows: int,
    max_samples: int,
    kanal_namen: tuple[str, ...],
    kanal_skalierung: tuple[float, ...] = (),
) -> pd.DataFrame:
    """Liest CSV plain Format (keine Zeitachse in der Datei).

    Erkennt automatisch ob Trennzeichen ',' oder ';' und ob Dezimalzeichen
    '.' oder ',' verwendet wird.
    Gibt DataFrame mit benannten Kanalspalten zurück, OHNE 'Zeit (ms)'.
    Zeitachse über build_time_axis() oder build_display_df() erzeugen.
    """
    n_kanäle = len(kanal_namen)
    nrows    = max_samples if max_samples > 0 else None
    sep, dec = _sniff_csv_params(file_bytes, skip_rows)

    probe      = pd.read_csv(io.BytesIO(file_bytes), sep=sep, decimal=dec, header=None, skiprows=skip_rows, nrows=1)
    first_cell = str(probe.iloc[0, 0]).strip()

    try:
        float(first_cell)
        df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, decimal=dec, header=None, skiprows=skip_rows, nrows=nrows)
        df = df.dropna(axis=1, how='all')
        erste_zeile = df.iloc[0]
        df = df[[c for c in df.columns if pd.notna(erste_zeile[c])]]
        data_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(data_cols) < n_kanäle:
            raise ValueError(
                f"CSV enthält nur {len(data_cols)} befüllte numerische Spalten, "
                f"aber {n_kanäle} Kanäle konfiguriert."
            )
        result_df = pd.DataFrame()
        for i, name in enumerate(kanal_namen):
            skale = kanal_skalierung[i] if i < len(kanal_skalierung) else 1.0
            result_df[name] = np.asarray(df[data_cols[i]].values, dtype=np.float64) * skale

    except ValueError as exc:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, decimal=dec, nrows=nrows)
        df = df.dropna(axis=1, how='all')
        erste_zeile = df.iloc[0]
        df = df[[c for c in df.columns if pd.notna(erste_zeile[c])]]
        sensor_cols = [c for c in df.columns if c != 'Zeit (s)']
        if len(sensor_cols) < n_kanäle:
            raise ValueError(
                f"CSV enthält nur {len(sensor_cols)} befüllte Messspalten, "
                f"aber {n_kanäle} Kanäle konfiguriert."
            ) from exc
        result_df = pd.DataFrame()
        for i, name in enumerate(kanal_namen):
            skale = kanal_skalierung[i] if i < len(kanal_skalierung) else 1.0
            result_df[name] = np.asarray(df[sensor_cols[i]].values, dtype=np.float64) * skale

    return result_df


def read_hubmessung_txt(
    file_bytes: bytes,
    max_samples: int,
    kanal_namen: tuple[str, ...],
) -> pd.DataFrame:
    """Liest TXT-Datei für Hubmessungen (TAB-getrennt, fester Header-Block).

    Gibt DataFrame mit 'Zeit (ms)' und benannten Kanalspalten zurück.
    """
    n_kanäle = len(kanal_namen)
    nrows    = max_samples if max_samples > 0 else None

    content = file_bytes.decode('utf-8', errors='ignore')
    lines   = content.splitlines()

    data_start_idx = -1
    for i, line in enumerate(lines):
        if "####Test Data####" in line:
            data_start_idx = i + 2
            break

    if data_start_idx == -1:
        raise ValueError("TXT-Datei enthält keinen gültigen '####Test Data####' Block.")

    filtered_lines: list[str] = []
    for line in lines[data_start_idx:]:
        if "####JSON Data####" in line:
            break
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        try:
            float(parts[0])
            filtered_lines.append(line)
        except (ValueError, IndexError):
            continue
        if nrows is not None and len(filtered_lines) >= nrows:
            break

    if not filtered_lines:
        raise ValueError("Keine gültigen numerischen Daten in der TXT-Datei gefunden.")

    df = pd.read_csv(io.StringIO('\n'.join(filtered_lines)), sep='\t', decimal='.', header=None)

    erste_zeile = df.iloc[0]
    df = df[[c for c in df.columns if pd.notna(erste_zeile[c])]]

    time_col    = df.columns[0]
    sensor_cols = df.columns[1:]

    if len(sensor_cols) == 0:
        raise ValueError("TXT-Datei enthält keine Sensordaten-Spalten.")

    # Mehr Kanäle konfiguriert als in der Datei vorhanden → auf Verfügbares kappen
    if n_kanäle > len(sensor_cols):
        kanal_namen = kanal_namen[:len(sensor_cols)]
        n_kanäle    = len(kanal_namen)

    result_df = pd.DataFrame()
    result_df['Zeit (ms)'] = df[time_col].values
    for i, name in enumerate(kanal_namen):
        result_df[name] = df[sensor_cols[i]].values

    return result_df


def read_oszilloskop_csv(
    file_bytes: bytes,
    max_samples: int,
    kanal_namen: tuple[str, ...],
    kanal_skalierung: tuple[float, ...],
) -> tuple[pd.DataFrame, float]:
    """Liest Oszilloskop-CSV-Format.

    Dateiformat:
      Zeile 1: x-axis,1,2,3           (Kanal-Nummern)
      Zeile 2: second,Ampere,Volt,Volt (Einheiten)
      Daten:   Zeit(s), Kanal1, ...    (wissenschaftliche Notation; anfangs leer)

    Zeilen mit leeren Kanal-Spalten werden übersprungen.
    Zeitachse wird auf t=0 beim ersten vollständigen Sample normiert.
    Die Zeiteinheit wird automatisch gewählt (s/ms/µs/ns) sodass
    die Anzeigewerte zwischen 0.02 und 800 liegen.

    Gibt (DataFrame, hz_faktor) zurück:
    - DataFrame: 'Zeit (ms)' (Werte in der gewählten Einheit) + Kanalspalten
    - hz_faktor: Umrechnungsfaktor s → Anzeigeeinheit (z.B. 1e6 für µs)
    """
    n_kanäle = len(kanal_namen)
    nrows    = max_samples if max_samples > 0 else None

    content = file_bytes.decode('utf-8', errors='ignore')
    lines   = content.splitlines()

    if len(lines) < 3:
        raise ValueError("Oszilloskop-CSV: Datei hat zu wenige Zeilen.")

    data_lines: list[str] = []
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if not parts:
            continue
        try:
            float(parts[0])
        except (ValueError, IndexError):
            continue
        if len(parts) > n_kanäle and all(p.strip() for p in parts[1:n_kanäle + 1]):
            data_lines.append(line)
        if nrows is not None and len(data_lines) >= nrows:
            break

    if not data_lines:
        raise ValueError("Oszilloskop-CSV: Keine vollständigen Datenpunkte gefunden.")

    df = pd.read_csv(io.StringIO('\n'.join(data_lines)), header=None, sep=',', decimal='.')

    n_datenspalten = df.shape[1] - 1
    if n_datenspalten < n_kanäle:
        raise ValueError(
            f"Oszilloskop-CSV enthält nur {n_datenspalten} Kanäle, "
            f"aber {n_kanäle} konfiguriert."
        )

    zeit_s  = df.iloc[:, 0].values.astype(float)
    max_s   = float(abs(zeit_s[-1] - zeit_s[0])) if len(zeit_s) > 1 else 1e-3
    hz_faktor, _ = wahl_zeiteinheit(max_s)
    zeit_display = (zeit_s - zeit_s[0]) * hz_faktor

    result_df = pd.DataFrame()
    result_df['Zeit (ms)'] = zeit_display
    for i, name in enumerate(kanal_namen):
        skale = kanal_skalierung[i] if i < len(kanal_skalierung) else 1.0
        result_df[name] = df.iloc[:, i + 1].values.astype(float) * skale

    return result_df, hz_faktor


# ===========================================================================
# RECHTECK-FIT (Huberkennung)
# ===========================================================================

def berechne_rechteck_fit(
    zeit: np.ndarray,
    signal: np.ndarray,
) -> dict | None:
    """Iterativer k-Means-Schwellwert-Finder für verrauschte Rechtecksignale.

    Gibt dict mit 'runs', 'y_low', 'y_high' zurück oder None wenn kein
    Rechteck erkennbar ist. Runs: Liste von {'t_start', 't_end'}-Pulsen.
    """
    if len(signal) == 0:
        return None
    valid = ~np.isnan(signal)
    if not np.any(valid):
        return None
    signal = signal[valid]
    zeit   = zeit[valid]

    min_val = float(np.nanpercentile(signal, 5))
    max_val = float(np.nanpercentile(signal, 95))
    if max_val <= min_val:
        return None

    threshold  = 0.5 * (min_val + max_val)
    low_center = min_val

    for _ in range(_RECHTECK_K_MEANS_ITER):
        high_mask = signal >= threshold
        low_mask  = signal < threshold
        if not np.any(high_mask) or not np.any(low_mask):
            break
        new_low  = float(np.median(signal[low_mask]))
        new_high = float(np.median(signal[high_mask]))
        if new_high <= new_low:
            break
        new_threshold = 0.5 * (new_low + new_high)
        low_center = new_low
        if np.isclose(new_threshold, threshold):
            threshold = new_threshold
            break
        threshold = new_threshold

    high_mask = signal >= threshold
    low_mask  = signal < threshold
    if not np.any(high_mask) or not np.any(low_mask):
        return None

    runs: list[dict] = []
    start = 0
    while start < len(high_mask):
        if high_mask[start]:
            end = start
            while end < len(high_mask) and high_mask[end]:
                end += 1
            runs.append({'t_start': float(zeit[start]), 't_end': float(zeit[end - 1])})
            start = end
        else:
            start += 1

    if not runs:
        return None
    high_center = float(np.median(signal[signal >= threshold]))
    return {'runs': runs, 'y_low': low_center, 'y_high': high_center}


# ===========================================================================
# HOCHRANGIGE API – EINHEITLICHER EINSTIEGSPUNKT
# ===========================================================================

def load_raw(
    file_bytes: bytes,
    file_type: str,
    skip_rows: int,
    max_samples: int,
    kanal_namen: tuple[str, ...],
    kanal_skalierung: tuple[float, ...] = (),
) -> tuple[pd.DataFrame, float]:
    """Liest Rohdaten formatunabhängig ein.

    Gibt (DataFrame, hz_faktor) zurück:
    - 'Hubmessung': hz_faktor = 1000.0 (Zeit bereits in ms)
    - 'Oszilloskop CSV': hz_faktor automatisch gewählt (1e3/1e6/1e9/1.0)
    - 'CSV plain': hz_faktor = 1000.0 (Zeitachse wird separat gebaut)
    """
    if file_type == "Hubmessung":
        return read_hubmessung_txt(file_bytes, max_samples, kanal_namen), 1000.0
    if file_type == "Oszilloskop CSV":
        return read_oszilloskop_csv(file_bytes, max_samples, kanal_namen, kanal_skalierung)
    return read_csv_plain(file_bytes, skip_rows, max_samples, kanal_namen, kanal_skalierung), 1000.0


def detect_kanal_count(file_bytes: bytes, file_type: str, skip_rows: int = 0) -> int:
    """Erkennt die Anzahl der Messdaten-Kanäle in der Datei (ohne Zeitspalte).

    Bei Fehler oder unbekanntem Format wird 4 zurückgegeben.
    """
    try:
        if file_type == "Oszilloskop CSV":
            kanal_indices, _ = peek_oszilloskop_header(file_bytes)
            n = len(kanal_indices)
            if n == 0:
                # Fallback für nicht-numerische Spaltenköpfe (z.B. CH1, CH2, ...)
                content = file_bytes.decode('utf-8', errors='ignore')
                first_line = content.splitlines()[0] if content.splitlines() else ''
                header_cols = [p.strip() for p in first_line.split(',')[1:] if p.strip()]
                n = len(header_cols)
            return max(1, n)

        if file_type == "Hubmessung":
            content = file_bytes.decode('utf-8', errors='ignore')
            lines   = content.splitlines()
            for i, line in enumerate(lines):
                if "####Test Data####" in line:
                    for raw in lines[i + 2:]:
                        raw = raw.strip()
                        if not raw:
                            continue
                        parts = raw.split('\t')
                        try:
                            float(parts[0])
                            return max(1, len([p for p in parts if p.strip()]) - 1)
                        except (ValueError, IndexError):
                            continue
            return 1

        # CSV plain
        sep, dec = _sniff_csv_params(file_bytes, skip_rows)
        probe      = pd.read_csv(io.BytesIO(file_bytes), sep=sep, decimal=dec, header=None,
                                 skiprows=skip_rows, nrows=3)
        probe      = probe.dropna(axis=1, how='all')
        first_cell = str(probe.iloc[0, 0]).strip()
        try:
            float(first_cell)
            data_cols = [c for c in probe.columns if pd.api.types.is_numeric_dtype(probe[c])]
            return max(1, len(data_cols))
        except ValueError:
            sensor_cols = [c for c in probe.columns if c != 'Zeit (s)']
            return max(1, len(sensor_cols))
    except Exception:
        return 4


def build_display_df(
    raw_df: pd.DataFrame,
    file_type: str,
    sample_rate_hz: float,
    kanal_namen: tuple[str, ...],
    offsets: tuple[float, ...],
    zeit_hz_faktor: float = 1000.0,
) -> tuple[pd.DataFrame, float]:
    """Erstellt den anzeigefertigen DataFrame mit Zeitachse und Y-Offsets.

    Für 'Hubmessung' und 'Oszilloskop CSV' wird die Zeitachse aus dem raw_df
    übernommen und die Samplerate daraus abgeleitet.
    Für 'CSV plain' wird die Zeitachse über build_time_axis() erzeugt.

    zeit_hz_faktor: Anzeigeeinheiten pro Sekunde (z.B. 1000 für ms, 1000000 für µs); sample_rate_hz = zeit_hz_faktor / dt.
    Gibt (df_full, tatsächliche_samplerate_hz) zurück.
    """
    if file_type in ("Hubmessung", "Oszilloskop CSV"):
        zeit = np.asarray(raw_df['Zeit (ms)'].values, dtype=np.float64)
        if len(zeit) > 1:
            dt = float(zeit[1] - zeit[0])
            if dt > 0:
                sample_rate_hz = zeit_hz_faktor / dt
    else:
        zeit = build_time_axis(len(raw_df), sample_rate_hz, hz_faktor=zeit_hz_faktor)

    kanal_arrays = tuple(raw_df[name].values for name in kanal_namen)
    df_full = apply_offsets(kanal_namen, kanal_arrays, offsets, zeit)
    return df_full, sample_rate_hz
