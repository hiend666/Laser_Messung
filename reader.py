"""
Datei-Einleser und Datenaufbereitung für Messdaten-Auswertung.
Unterstützte Formate: CSV plain, Hubmessung TXT, Oszilloskop CSV.
Kein Streamlit – nur Python/NumPy/Pandas/SciPy.
"""
from __future__ import annotations
import io
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# Polynomgrad für alle Savitzky-Golay-Filter
SAVGOL_POLYNOM = 3


# ===========================================================================
# ZEITACHSE
# ===========================================================================

def build_time_axis(n_samples: int, sr_hz: float) -> np.ndarray:
    """Zeitvektor in ms für n_samples bei Abtastrate sr_hz (Hz)."""
    return np.arange(n_samples) * (1000.0 / sr_hz)


# ===========================================================================
# SAVITZKY-GOLAY ABLEITUNGEN
# ===========================================================================

def clamp_savgol_fenster(fenster: int, n: int) -> int:
    """Klemmt SG-Fenstergröße auf gültigen Wert (ungerade, ≥ 5, < n)."""
    if fenster >= n:
        fenster = n if n % 2 == 1 else n - 1
    if fenster % 2 == 0:
        fenster -= 1
    return fenster


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
        data[name] = arr + off
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


def read_csv_plain(
    file_bytes: bytes,
    skip_rows: int,
    max_samples: int,
    kanal_namen: tuple[str, ...],
) -> pd.DataFrame:
    """Liest CSV plain Format (Komma-getrennt, keine Zeitachse in der Datei).

    Gibt DataFrame mit benannten Kanalspalten zurück, OHNE 'Zeit (ms)'.
    Zeitachse über build_time_axis() oder build_display_df() erzeugen.
    """
    n_kanäle = len(kanal_namen)
    nrows    = max_samples if max_samples > 0 else None

    probe      = pd.read_csv(io.BytesIO(file_bytes), sep=',', decimal='.', header=None, skiprows=skip_rows, nrows=1)
    first_cell = str(probe.iloc[0, 0]).strip()

    try:
        float(first_cell)
        df = pd.read_csv(io.BytesIO(file_bytes), sep=',', decimal='.', header=None, skiprows=skip_rows, nrows=nrows)
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
            result_df[name] = df[data_cols[i]].values

    except ValueError as exc:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=',', decimal='.', nrows=nrows)
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
            result_df[name] = df[sensor_cols[i]].values

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

    if len(sensor_cols) < n_kanäle:
        raise ValueError(
            f"TXT-Datei enthält nur {len(sensor_cols)} Sensordaten-Spalten, "
            f"aber {n_kanäle} Kanäle konfiguriert."
        )

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
) -> pd.DataFrame:
    """Liest Oszilloskop-CSV-Format.

    Dateiformat:
      Zeile 1: x-axis,1,2,3           (Kanal-Nummern)
      Zeile 2: second,Ampere,Volt,Volt (Einheiten)
      Daten:   Zeit(s), Kanal1, ...    (wissenschaftliche Notation; anfangs leer)

    Zeilen mit leeren Kanal-Spalten werden übersprungen.
    Zeitachse wird auf t=0 beim ersten vollständigen Sample normiert.

    Gibt DataFrame mit 'Zeit (ms)' und benannten, skalierten Kanalspalten zurück.
    """
    n_kanäle = len(kanal_namen)
    nrows    = max_samples if max_samples > 0 else None

    content = file_bytes.decode('utf-8', errors='ignore')
    lines   = content.splitlines()

    if len(lines) < 3:
        raise ValueError("Oszilloskop-CSV: Datei hat zu wenige Zeilen.")

    n_cols = len(lines[0].split(','))   # x-axis + n Kanäle

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
        kanal_parts = parts[1:n_cols]
        if len(kanal_parts) >= n_kanäle and all(p.strip() for p in kanal_parts[:n_kanäle]):
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
    zeit_ms = (zeit_s - zeit_s[0]) * 1000.0

    result_df = pd.DataFrame()
    result_df['Zeit (ms)'] = zeit_ms
    for i, name in enumerate(kanal_namen):
        skale = kanal_skalierung[i] if i < len(kanal_skalierung) else 1.0
        result_df[name] = df.iloc[:, i + 1].values.astype(float) * skale

    return result_df


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
) -> pd.DataFrame:
    """Liest Rohdaten formatunabhängig ein.

    Gibt DataFrame zurück:
    - 'Hubmessung' / 'Oszilloskop CSV': enthält 'Zeit (ms)' + Kanalspalten
    - 'CSV plain': enthält nur Kanalspalten (keine Zeitachse)
    """
    if file_type == "Hubmessung":
        return read_hubmessung_txt(file_bytes, max_samples, kanal_namen)
    if file_type == "Oszilloskop CSV":
        return read_oszilloskop_csv(file_bytes, max_samples, kanal_namen, kanal_skalierung)
    return read_csv_plain(file_bytes, skip_rows, max_samples, kanal_namen)


def build_display_df(
    raw_df: pd.DataFrame,
    file_type: str,
    sample_rate_hz: float,
    kanal_namen: tuple[str, ...],
    offsets: tuple[float, ...],
) -> tuple[pd.DataFrame, float]:
    """Erstellt den anzeigefertigen DataFrame mit Zeitachse und Y-Offsets.

    Für 'Hubmessung' und 'Oszilloskop CSV' wird die Zeitachse aus dem raw_df
    übernommen und die Samplerate daraus abgeleitet.
    Für 'CSV plain' wird die Zeitachse über build_time_axis() erzeugt.

    Gibt (df_full, tatsächliche_samplerate_hz) zurück.
    """
    if file_type in ("Hubmessung", "Oszilloskop CSV"):
        zeit = raw_df['Zeit (ms)'].values
        if len(zeit) > 1:
            dt_ms = float(zeit[1] - zeit[0])
            if dt_ms > 0:
                sample_rate_hz = 1000.0 / dt_ms
    else:
        zeit = build_time_axis(len(raw_df), sample_rate_hz)

    kanal_arrays = tuple(raw_df[name].values for name in kanal_namen)
    df_full = apply_offsets(kanal_namen, kanal_arrays, offsets, zeit)
    return df_full, sample_rate_hz
