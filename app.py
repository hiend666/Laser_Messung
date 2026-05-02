# pip install streamlit pandas plotly scipy "kaleido==0.2.1" reportlab
# streamlit run app.py
"""
Messdaten-Auswertung – Laservibrometer CSV.

Zeigt Weg-, Geschwindigkeits- und Beschleunigungskurven und
exportiert Ergebnisse als PDF oder PNG.
"""
import io
import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import savgol_filter
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

VERSION = "v1.00.06"

# ---------------------------------------------------------------------------
# KONSTANTEN
# ---------------------------------------------------------------------------

V_ACHSE_LIMIT_MM_S = 3_200    # Feste Y-Grenze Geschwindigkeitsachse  ± mm/s
A_ACHSE_LIMIT_M_S2 = 20_000   # Feste Y-Grenze Beschleunigungsachse   ± m/s²

MAX_PLOT_PUNKTE    = 5_000     # Downsampling-Schwelle für interaktives Diagramm
SAVGOL_POLYNOM     = 3         # Polynomgrad für alle Savitzky-Golay-Filter

# Diagramm-Farben – Kanäle
FARBE_KANAL1    = '#003366'
FARBE_KANAL2    = '#4c78a8'
FARBE_KANAL3    = '#d62728'
FARBE_KANAL4    = '#2ca02c'
KANAL_FARBEN    = [FARBE_KANAL1, FARBE_KANAL2, FARBE_KANAL3, FARBE_KANAL4]

# Diagramm-Farben – Auswertung
FARBE_GESCHW    = 'purple'
FARBE_BESCHL    = 'orange'
FARBE_V_SCHNITT = 'green'
FARBE_VMAX      = 'red'
FARBE_AMAX      = 'orange'
FARBE_CURSOR    = 'red'
FARBE_RECHTECK  = 'lime'

# ---------------------------------------------------------------------------
# STREAMLIT-SEITENKONFIGURATION UND CSS
# ---------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Messdaten Auswertung")
st.markdown("""
    <style>
    div[data-baseweb="slider"] > div > div > div {
        background-color: #e6e6e6 !important;
    }
    div[data-baseweb="slider"] [role="slider"] {
        background-color: #003366 !important;
        border: 1px solid #002244 !important;
    }
    div[data-baseweb="slider"] [data-testid="stTickBar"] {
        background-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SESSION STATE – INITIALISIERUNG
# ---------------------------------------------------------------------------
# Zwei-Key-Muster: freie Keys (xa, xb, off1, …) sind die Wahrheitsquelle.
# Widget-Keys (xa_sw, xa_nw, …) gehören ausschließlich den Widgets;
# ihre on_change-Callbacks schreiben in den freien Key zurück.
# Externe Setter (Buttons, Auto-Reset) schreiben nur in freie Keys.
defaults = {
    'off1': 0.0,
    'off2': 0.0,
    'off3': 0.0,
    'off4': 0.0,
    'off1_slider': 0.0,
    'off2_slider': 0.0,
    'off3_slider': 0.0,
    'off4_slider': 0.0,
    'xa': 0.0,        # freie Wahrheitsquelle – nie Widget-Key
    'xb': 0.001,      # freie Wahrheitsquelle – nie Widget-Key
    'xa_sw': 0.0,     # Widget-Key: Slider XA
    'xb_sw': 0.001,   # Widget-Key: Slider XB
    'xa_nw': 0.0,     # Widget-Key: number_input XA
    'xb_nw': 0.001,   # Widget-Key: number_input XB
    'zoom_token': 0,
    'last_file_name': None,
    'sample_rate': 2.55,
    'sample_rate_unit': 'µs',
    'sample_rate_unit_toggle': True,
    'skip_rows': 12,
    'ch1_name': 'Festo',
    'ch2_name': 'DST',
    'ch3_name': '',   # leer = Kanal nicht einlesen
    'ch4_name': '',   # leer = Kanal nicht einlesen
    'max_samples': 8000,
    # Crop-State: None = "Show All", sonst t_start / t_end als float
    'crop_start': None,
    'crop_end': None,
    'show_velocity': False,
    'window_length': 30,
    'show_acceleration': False,
    'window_length_accel': 40,
    'sop_percent': 80,
    'v_axis_limit': 3_200,
    'a_axis_limit': 20_000,
    'sub_dateityp':   False,
    'sub_einlesen':   False,
    'sub_kanaele':    False,
    'sub_offsets':    False,
    'sub_grenzwerte': False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# HILFSFUNKTIONEN – SAVITZKY-GOLAY
# ---------------------------------------------------------------------------

def _clamp_savgol_fenster(fenster: int, n: int) -> int:
    """Klemmt Fenstergröße auf gültigen Wert für savgol_filter (ungerade, ≥ 5, < n)."""
    if fenster >= n:
        fenster = n if n % 2 == 1 else n - 1
    if fenster % 2 == 0:
        fenster -= 1
    return fenster


def _berechne_sg_ableitung(
    signal: np.ndarray, dt_s: float, fenster: int, ordnung: int
) -> np.ndarray | None:
    """Savitzky-Golay-Ableitung beliebiger Ordnung. Gibt None zurück wenn Signal zu kurz.

    ordnung 1 → Geschwindigkeit (signal_einheit / s)
    ordnung 2 → Beschleunigung  (signal_einheit / s²)
    """
    fenster = _clamp_savgol_fenster(fenster, len(signal))
    if fenster < 5:
        return None
    return savgol_filter(signal, fenster, SAVGOL_POLYNOM, deriv=ordnung, delta=dt_s, mode='mirror')


# ---------------------------------------------------------------------------
# GECACHTE DATENFUNKTIONEN
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(
    file_bytes: bytes,
    skip_rows: int,
    max_samples: int,
    kanal_namen: tuple[str, ...],
    file_type: str,
) -> pd.DataFrame:
    """Liest Daten im CSV- oder TXT-Format und gibt DataFrame mit den konfigurierten Kanälen zurück.

    Leere Spalten (NaN in der ersten Datenzeile) werden vor der Kanalzuweisung herausgefiltert.
    kanal_namen bestimmt, wie viele Spalten eingelesen werden und wie sie heißen.
    """
    n_kanäle = len(kanal_namen)
    nrows    = max_samples if max_samples > 0 else None

    if file_type == "Hubmessung":
        # TXT-Datei für Hubmessung: TAB-getrennt, Header-Blöcke überspringen
        content = file_bytes.decode('utf-8', errors='ignore')

        # Finde den Beginn der Daten nach "####Test Data####"
        # splitlines() verarbeitet \n, \r\n und \r korrekt
        lines = content.splitlines()
        data_start_idx = -1
        for i, line in enumerate(lines):
            if "####Test Data####" in line:
                # Daten beginnen zwei Zeilen später (nach "Stroke Data" und Header)
                data_start_idx = i + 2
                break

        if data_start_idx == -1:
            raise ValueError("TXT-Datei enthält keinen gültigen '####Test Data####' Block.")

        # Daten ab data_start_idx einlesen
        data_lines = lines[data_start_idx:]
        # Leere Zeilen und nicht-numerische Zeilen filtern
        filtered_lines = []
        for line in data_lines:
            line = line.strip()
            if not line:
                continue
            # Prüfe ob die Zeile numerische Daten enthält
            parts = line.split('\t')
            try:
                float(parts[0])  # Erste Spalte sollte Zeit sein
                filtered_lines.append(line)
            except (ValueError, IndexError):
                continue

        if not filtered_lines:
            raise ValueError("Keine gültigen numerischen Daten in der TXT-Datei gefunden.")

        # Als CSV-String behandeln
        data_content = '\n'.join(filtered_lines)
        df = pd.read_csv(
            io.StringIO(data_content), sep='\t', decimal='.', header=None
        )

        # Spalten herausfiltern, die in der ersten Datenzeile leer sind
        erste_zeile = df.iloc[0]
        df = df[[c for c in df.columns if pd.notna(erste_zeile[c])]]

        # Erste Spalte ist Zeit in ms, restliche sind Sensordaten
        time_col = df.columns[0]
        sensor_cols = df.columns[1:]

        if len(sensor_cols) < n_kanäle:
            raise ValueError(
                f"TXT-Datei enthält nur {len(sensor_cols)} Sensordaten-Spalten, "
                f"aber {n_kanäle} Kanäle sind konfiguriert."
            )

        # DataFrame mit Zeit und Sensordaten erstellen
        result_df = pd.DataFrame()
        result_df['Zeit (ms)'] = df[time_col]
        for i, kanal_name in enumerate(kanal_namen):
            result_df[kanal_name] = df[sensor_cols[i]]

        return result_df

    else:
        # CSV plain Format
        probe      = pd.read_csv(
            io.BytesIO(file_bytes), sep=',', decimal='.', header=None, skiprows=skip_rows, nrows=1
        )
        first_cell = str(probe.iloc[0, 0]).strip()

        try:
            float(first_cell)
            # Roh-Format: keine Spaltenköpfe, erste Zelle ist bereits numerisch
            df = pd.read_csv(
                io.BytesIO(file_bytes), sep=',', decimal='.', header=None,
                skiprows=skip_rows, nrows=nrows
            )
            df = df.dropna(axis=1, how='all')
            # Spalten herausfiltern, die in der ersten Datenzeile leer sind
            erste_zeile = df.iloc[0]
            df = df[[c for c in df.columns if pd.notna(erste_zeile[c])]]
            data_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(data_cols) < n_kanäle:
                raise ValueError(
                    f"CSV enthält nur {len(data_cols)} befüllte numerische Spalten, "
                    f"aber {n_kanäle} Kanäle sind konfiguriert."
                )
            # DataFrame mit berechneten Zeitstempeln und Sensordaten erstellen
            result_df = pd.DataFrame()
            result_df['Zeit (ms)'] = build_time_axis(len(df), sample_rate)
            for i, kanal_name in enumerate(kanal_namen):
                result_df[kanal_name] = df[data_cols[i]]

        except ValueError as exc:
            # Sauber-Format: erste Zeile ist Spaltenheader
            df = pd.read_csv(io.BytesIO(file_bytes), sep=',', decimal='.', nrows=nrows)
            df = df.dropna(axis=1, how='all')
            # Spalten herausfiltern, die in der ersten Datenzeile leer sind
            erste_zeile = df.iloc[0]
            df = df[[c for c in df.columns if pd.notna(erste_zeile[c])]]
            sensor_cols = [c for c in df.columns if c != 'Zeit (s)']
            if len(sensor_cols) < n_kanäle:
                raise ValueError(
                    f"CSV enthält nur {len(sensor_cols)} befüllte Messspalten, "
                    f"aber {n_kanäle} Kanäle sind konfiguriert."
                ) from exc
            # DataFrame mit Zeit und Sensordaten erstellen
            result_df = pd.DataFrame()
            result_df['Zeit (ms)'] = build_time_axis(len(df), sample_rate)
            for i, kanal_name in enumerate(kanal_namen):
                result_df[kanal_name] = df[sensor_cols[i]]

        return result_df


@st.cache_data
def build_time_axis(n_samples: int, sr: float) -> np.ndarray:
    """Erzeugt Zeitvektor in ms für n_samples bei Samplerate sr (Hz)."""
    return np.arange(n_samples) * (1000.0 / sr)


@st.cache_data
def apply_offsets(
    kanal_namen: tuple[str, ...],
    kanal_arrays: tuple,        # tuple of np.ndarray, je Kanal ein Array
    offsets: tuple[float, ...],
    zeit: np.ndarray,
) -> pd.DataFrame:
    """Erzeugt den verarbeiteten DataFrame – nur bei echten Änderungen neu berechnet."""
    data: dict = {'Zeit (ms)': zeit}
    for name, arr, off in zip(kanal_namen, kanal_arrays, offsets):
        data[name] = arr + off
    return pd.DataFrame(data)


@st.cache_data
def compute_best_fit_rectangle(zeit: np.ndarray, signal: np.ndarray):
    """Iteratives Rechteck-Fit für verrauschte Rechtecksignale (Huberkennung).

    Gibt dict mit 'runs' (Liste von Pulsen mit t_start/t_end), 'y_low' und
    'y_high' zurück, oder None wenn kein Rechteck erkennbar.
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

    threshold   = 0.5 * (min_val + max_val)
    low_center  = min_val
    high_center = max_val

    # k-Means-ähnliche Iteration für robusten Schwellwert
    for _ in range(5):
        high_mask = signal >= threshold
        low_mask  = signal < threshold
        if not np.any(high_mask) or not np.any(low_mask):
            break
        new_low  = float(np.median(signal[low_mask]))
        new_high = float(np.median(signal[high_mask]))
        if new_high <= new_low:
            break
        new_threshold = 0.5 * (new_low + new_high)
        low_center  = new_low
        high_center = new_high
        if np.isclose(new_threshold, threshold):
            threshold = new_threshold
            break
        threshold = new_threshold

    high_mask = signal >= threshold
    low_mask  = signal < threshold
    if not np.any(high_mask) or not np.any(low_mask):
        return None

    # Zusammenhängende High-Runs (Pulse) ermitteln
    runs  = []
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
    return {'runs': runs, 'y_low': low_center, 'y_high': high_center}


# ---------------------------------------------------------------------------
# CALLBACKS – Zwei-Key-Muster
# Widgets schreiben immer in den freien Key (xa/xb/off1…4), nie umgekehrt.
# ---------------------------------------------------------------------------

def update_xa_from_slider():
    st.session_state.xa = max(0.0, float(st.session_state.xa_sw))

def update_xa_from_num():
    st.session_state.xa = max(0.0, float(st.session_state.xa_nw))

def update_xb_from_slider():
    st.session_state.xb = max(0.0, float(st.session_state.xb_sw))

def update_xb_from_num():
    st.session_state.xb = max(0.0, float(st.session_state.xb_nw))

def update_off1_from_slider():
    st.session_state.off1 = st.session_state.off1_slider

def update_off2_from_slider():
    st.session_state.off2 = st.session_state.off2_slider

def update_off3_from_slider():
    st.session_state.off3 = st.session_state.off3_slider

def update_off4_from_slider():
    st.session_state.off4 = st.session_state.off4_slider

# Indizierter Zugriff auf Offset-Callbacks (Index 0–3 entspricht Kanal 1–4)
OFF_CALLBACKS = [
    update_off1_from_slider,
    update_off2_from_slider,
    update_off3_from_slider,
    update_off4_from_slider,
]

def update_sample_rate_unit():
    new_unit = "µs" if st.session_state.sample_rate_unit_toggle else "Hz"
    old_unit = st.session_state.sample_rate_unit
    if new_unit != old_unit:
        if st.session_state.sample_rate > 0:
            st.session_state.sample_rate = 1_000_000.0 / st.session_state.sample_rate
        st.session_state.sample_rate_unit = new_unit


def update_sample_rate_for_file_type():
    """Setzt die Samplerate automatisch basierend auf dem Dateityp."""
    if st.session_state.get('file_type_radio', 'CSV plain') == "Hubmessung":
        # Hubmessungen haben feste Samplerate von 2.55 µs
        st.session_state.sample_rate = 2.55
        st.session_state.sample_rate_unit = "µs"
        st.session_state.sample_rate_unit_toggle = True
        st.session_state.ch1_name = 'Hub'
        st.session_state.ch2_name = ''
        st.session_state.ch3_name = ''
        st.session_state.ch4_name = ''
    # Für CSV plain bleiben die Einstellungen wie sie sind


# ---------------------------------------------------------------------------
# HILFSFUNKTIONEN – INDEX UND GESCHWINDIGKEIT
# ---------------------------------------------------------------------------

def get_idx_at_x(x_ms: float, sample_rate: float, max_idx: int) -> int:
    """Wandelt Zeitwert (ms) in DataFrame-Index um. O(1)."""
    return int(np.clip(round(x_ms / 1000.0 * sample_rate), 0, max_idx))


def _berechne_ableitungen_fuer_diagramm(
    df_quelle: pd.DataFrame,
    sensor: str,
    show_velocity: bool,
    show_acceleration: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Berechnet Geschwindigkeit (mm/s) und Beschleunigung (m/s²) für die Diagramm-Darstellung.

    Gibt (velocity, acceleration) zurück, jeweils None wenn nicht aktiviert oder zu wenig Daten.
    """
    if len(df_quelle) <= 1:
        return None, None

    arr  = df_quelle[sensor].values
    dt_s = (df_quelle['Zeit (ms)'].iloc[1] - df_quelle['Zeit (ms)'].iloc[0]) / 1000.0

    velocity = None
    if show_velocity:
        roh = _berechne_sg_ableitung(arr, dt_s, st.session_state.window_length, 1)
        velocity = roh / 1000.0 if roh is not None else None          # µm/s → mm/s

    acceleration = None
    if show_acceleration:
        roh = _berechne_sg_ableitung(arr, dt_s, st.session_state.window_length_accel, 2)
        acceleration = roh / 1_000_000.0 if roh is not None else None  # µm/s² → m/s²

    return velocity, acceleration


def _zeichne_rechteck_fit(
    fig: go.Figure,
    rect_fit: dict,
    bereich_min: float,
    bereich_max: float,
    mit_fuellung: bool,
):
    """Fügt Rechteck-Fit-Traces und optionale Füllformen zum Diagramm hinzu."""
    for idx, run in enumerate(rect_fit['runs']):
        clipped_start = max(run['t_start'], bereich_min)
        clipped_end   = min(run['t_end'],   bereich_max)
        if clipped_start >= clipped_end:
            continue
        fig.add_trace(go.Scatter(
            x=[clipped_start, clipped_end],
            y=[rect_fit['y_high'], rect_fit['y_high']],
            mode='lines',
            name='Rechteck-Fit' if idx == 0 else None,
            showlegend=(idx == 0),
            line=dict(color=FARBE_RECHTECK, dash='dash', width=2),
        ))
        if mit_fuellung:
            for x_kante in (clipped_start, clipped_end):
                fig.add_shape(
                    type='line',
                    x0=x_kante, x1=x_kante,
                    y0=rect_fit['y_low'], y1=rect_fit['y_high'],
                    line=dict(color=FARBE_RECHTECK, width=1, dash='dash'),
                )
            fig.add_shape(
                type='rect',
                x0=clipped_start, x1=clipped_end,
                y0=rect_fit['y_low'], y1=rect_fit['y_high'],
                line=dict(width=0),
                fillcolor='rgba(0,255,0,0.08)',
            )


# ---------------------------------------------------------------------------
# HILFSFUNKTION – SPEED ON POINT
# ---------------------------------------------------------------------------

def _finde_sop_kreuzungen(
    zeit: np.ndarray,
    signal: np.ndarray,
    rect_fit: dict,
    sop_percent: float,
    sample_rate: float,
    halbes_zeitfenster: int,
) -> tuple[list, float]:
    """Findet SOP-Punkte an steigenden Flanken des Rechteck-Fits.

    Gibt (sop_linien, v_sop) zurück:
    - sop_linien: Liste von (t_sop, t_links, t_rechts, y_level) für Diagramm-Linien
    - v_sop:      Geschwindigkeit am ersten Kreuzungspunkt (mm/s), oder nan
    """
    hub = rect_fit['y_high'] - rect_fit['y_low']
    if hub <= 0:
        return [], float('nan')

    sop_level = rect_fit['y_low'] + (sop_percent / 100.0) * hub
    n         = len(signal)
    ergebnisse = []

    for run in rect_fit['runs']:
        # Suchfenster: kurz vor Pulsstart bis ins erste Drittel des Pulses
        t_suche_start = run['t_start'] - 0.5
        t_suche_ende  = run['t_start'] + max(0.1, (run['t_end'] - run['t_start']) * 0.3)
        idx_fenster   = np.where((zeit >= t_suche_start) & (zeit <= t_suche_ende))[0]
        if len(idx_fenster) < 2:
            continue

        s             = signal[idx_fenster]
        kreuzungs_pos = np.where((s[:-1] < sop_level) & (s[1:] >= sop_level))[0]
        if len(kreuzungs_pos) == 0:
            continue

        abs_idx = int(idx_fenster[kreuzungs_pos[0] + 1])

        # Geschwindigkeit an der Kreuzung (finite difference über halbes_zeitfenster)
        i0    = max(0, abs_idx - halbes_zeitfenster)
        i1    = min(n - 1, abs_idx + halbes_zeitfenster)
        dt_s  = (i1 - i0) / sample_rate
        v_sop = ((signal[i1] - signal[i0]) / 1000.0) / dt_s if dt_s > 0 else float('nan')

        # Linie: je 10 Samples links und rechts des Kreuzungspunkts
        t_sop    = float(zeit[abs_idx])
        t_links  = float(zeit[max(0, abs_idx - 10)])
        t_rechts = float(zeit[min(n - 1, abs_idx + 10)])
        ergebnisse.append((t_sop, t_links, t_rechts, sop_level, v_sop))

    if not ergebnisse:
        return [], float('nan')

    # Format pro Eintrag: (t_sop, t_links, t_rechts, y_level)
    sop_linien = [(t_sop, t0, t1, y) for t_sop, t0, t1, y, _ in ergebnisse]
    v_sop_wert = ergebnisse[0][4]   # Geschwindigkeit am ersten Kreuzungspunkt
    return sop_linien, v_sop_wert


# ---------------------------------------------------------------------------
# EXPORT: DIAGRAMM ALS PNG
# ---------------------------------------------------------------------------

def build_chart_png(
    df,
    sensor_namen: list[str],
    active_sensor: str,
    xa, xb, ya, yb, show_v_avg,
    t_vmax_start, y_vmax_start, t_vmax_ende, y_vmax_ende, has_vmax,
    t_amax_falling, y_amax_falling, has_amax_falling,
    t_amax_rising,  y_amax_rising,  has_amax_rising,
    show_rect_fit=False, rect_fit=None,
    show_velocity=False, window_length=21,
    show_acceleration=False, window_length_accel=21,
    sop_linien=None,
) -> bytes:
    """Rendert das Diagramm mit Kaleido zu PNG-Bytes für den Export."""

    # Y-Achse: 15 % Puffer über Signalbereich damit Legende den Graph nicht verdeckt
    y_max_e   = float(df[sensor_namen].max().max())
    y_min_e   = float(df[sensor_namen].min().min())
    y_range_e = [y_min_e, y_max_e + (y_max_e - y_min_e) * 0.15]

    # Ableitungen für Export-Diagramm berechnen
    velocity = acceleration = None
    if len(df) > 1:
        arr  = df[active_sensor].values
        dt_s = (df['Zeit (ms)'].iloc[1] - df['Zeit (ms)'].iloc[0]) / 1000.0
        if show_velocity:
            roh = _berechne_sg_ableitung(arr, dt_s, window_length, 1)
            velocity = roh / 1000.0 if roh is not None else None
        if show_acceleration:
            roh = _berechne_sg_ableitung(arr, dt_s, window_length_accel, 2)
            acceleration = roh / 1_000_000.0 if roh is not None else None

    export_fig = go.Figure()

    for i, name in enumerate(sensor_namen):
        export_fig.add_trace(go.Scatter(
            x=df['Zeit (ms)'], y=df[name],
            name=name, line=dict(color=KANAL_FARBEN[i]),
        ))

    export_fig.add_vline(x=xa, line_dash="dash", line_color=FARBE_CURSOR)
    export_fig.add_vline(x=xb, line_dash="dash", line_color=FARBE_CURSOR)

    if show_v_avg:
        export_fig.add_trace(go.Scatter(
            x=[xa, xb], y=[ya, yb],
            mode='lines+markers', name='v-Schnitt',
            line=dict(color=FARBE_V_SCHNITT, width=2, dash='dot'),
        ))

    if rect_fit is not None:
        _zeichne_rechteck_fit(
            export_fig, rect_fit,
            df['Zeit (ms)'].min(), df['Zeit (ms)'].max(),
            mit_fuellung=show_rect_fit,
        )

    if has_vmax and t_vmax_start is not None:
        export_fig.add_trace(go.Scatter(
            x=[t_vmax_start, t_vmax_ende], y=[y_vmax_start, y_vmax_ende],
            mode='lines+markers', name='v-max',
            line=dict(color=FARBE_VMAX, width=2),
        ))
    if has_amax_falling and t_amax_falling is not None:
        export_fig.add_trace(go.Scatter(
            x=[t_amax_falling], y=[y_amax_falling],
            mode='markers', name='a-max',
            marker=dict(color=FARBE_AMAX, size=14, symbol='cross',
                        line=dict(color=FARBE_AMAX, width=2)),
        ))
    if has_amax_rising and t_amax_rising is not None:
        export_fig.add_trace(go.Scatter(
            x=[t_amax_rising], y=[y_amax_rising],
            mode='markers', name='a-min',
            marker=dict(color=FARBE_AMAX, size=12, symbol='circle',
                        line=dict(color=FARBE_AMAX, width=2)),
        ))
    if sop_linien:
        t_min_export   = float(df['Zeit (ms)'].min())
        t_max_export   = float(df['Zeit (ms)'].max())
        erste_sichtbar = True
        for t_sop, t0, t1, y_lvl in sop_linien:
            if not (t_min_export <= t_sop <= t_max_export):
                continue
            export_fig.add_trace(go.Scatter(
                x=[max(t0, t_min_export), min(t1, t_max_export)], y=[y_lvl, y_lvl],
                mode='lines',
                name='SOP' if erste_sichtbar else None,
                showlegend=erste_sichtbar,
                line=dict(color=FARBE_GESCHW, width=2),
            ))
            export_fig.add_trace(go.Scatter(
                x=[t_sop], y=[y_lvl],
                mode='markers', showlegend=False,
                marker=dict(color=FARBE_GESCHW, size=14, symbol='x',
                            line=dict(color=FARBE_GESCHW, width=2)),
            ))
            erste_sichtbar = False
    if show_velocity and velocity is not None:
        export_fig.add_trace(go.Scatter(
            x=df['Zeit (ms)'], y=velocity,
            name='Geschwindigkeit', yaxis='y2', line=dict(color=FARBE_GESCHW),
        ))
    if show_acceleration and acceleration is not None:
        export_fig.add_trace(go.Scatter(
            x=df['Zeit (ms)'], y=acceleration,
            name='Beschleunigung', yaxis='y3', line=dict(color=FARBE_BESCHL),
        ))

    export_fig.update_layout(
        xaxis_title="Zeit (ms)",
        yaxis_title="Weg (µm)",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
        xaxis=dict(autorange=True, rangemode='nonnegative'),
        yaxis=dict(range=y_range_e),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    if show_velocity and velocity is not None:
        export_fig.update_layout(
            yaxis2=dict(
                title='Geschwindigkeit (mm/s)',
                overlaying='y', side='right', showgrid=False,
                range=[-st.session_state.v_axis_limit, st.session_state.v_axis_limit],
            )
        )
    if show_acceleration and acceleration is not None:
        export_fig.update_layout(
            yaxis3=dict(
                title='Beschleunigung (m/s²)',
                overlaying='y', side='right', showgrid=False,
                position=0.85 if show_velocity else 1.0,
                range=[-st.session_state.a_axis_limit, st.session_state.a_axis_limit],
            )
        )
    return export_fig.to_image(format="png", width=1600, height=500, scale=2)


# ---------------------------------------------------------------------------
# EXPORT: PDF
# ---------------------------------------------------------------------------

def build_pdf(filename: str, chart_png: bytes, metrics: dict) -> bytes:
    """Erstellt ein A4-Querformat-PDF mit Diagramm und Kenngrößen-Tabelle."""
    buf      = io.BytesIO()
    page     = landscape(A4)       # 297 × 210 mm
    usable_w = page[0] - 30 * mm  # 267 mm (je 15 mm Rand)
    doc = SimpleDocTemplate(
        buf, pagesize=page,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=12*mm, bottomMargin=12*mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'ExportTitle', parent=styles['Normal'],
        fontSize=13, textColor=colors.HexColor('#003366'),
        fontName='Helvetica-Bold', spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        'ExportSub', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#666666'), spaceAfter=6,
    )
    ts_style = ParagraphStyle(
        'ExportTS', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#666666'),
        fontName='Helvetica', alignment=2,  # 2 = RIGHT
    )
    dt_now = datetime.datetime.now().strftime("%d.%m.%Y  %H:%M:%S")
    header_tbl = Table(
        [[Paragraph("Messdaten-Auswertung", title_style), Paragraph(dt_now, ts_style)]],
        colWidths=[usable_w * 0.7, usable_w * 0.3],
        rowHeights=[8*mm],
    )
    header_tbl.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
    ]))

    story = [
        header_tbl,
        Paragraph(f"Datei: {filename}", sub_style),
        Image(io.BytesIO(chart_png), width=usable_w, height=usable_w * 0.38),  # ~16:6 Ratio
        Spacer(1, 4*mm),
    ]

    # Kenngrößen auf zwei gleichbreite Reihen aufteilen
    items               = list(metrics.items())
    halb                = (len(items) + 1) // 2
    kenngroessen_oben   = items[:halb]
    kenngroessen_unten  = items[halb:]
    while len(kenngroessen_unten) < halb:
        kenngroessen_unten.append(("", ""))

    col_widths = [usable_w / halb] * halb

    def _make_kenngroessen_tabelle(zeilen_items):
        """Baut eine zweizeilige Kenngrößen-Tabelle (Label oben, Wert unten)."""
        labels = [k for k, _ in zeilen_items]
        values = [v for _, v in zeilen_items]
        tbl = Table([labels, values], colWidths=col_widths, rowHeights=[7*mm, 8*mm])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',  (0, 0), (-1, 0), colors.HexColor('#003366')),
            ('TEXTCOLOR',   (0, 0), (-1, 0), colors.white),
            ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',    (0, 0), (-1, 0), 8),
            ('ALIGN',       (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND',  (0, 1), (-1, 1), colors.HexColor('#f0f4f8')),
            ('FONTNAME',    (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE',    (0, 1), (-1, 1), 9),
            ('TEXTCOLOR',   (0, 1), (-1, 1), colors.HexColor('#003366')),
            ('GRID',        (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ]))
        return tbl

    story.append(_make_kenngroessen_tabelle(kenngroessen_oben))
    story.append(Spacer(1, 2*mm))
    story.append(_make_kenngroessen_tabelle(kenngroessen_unten))
    doc.build(story)
    buf.seek(0)
    return buf.read()


# ===========================================================================
# HAUPTBEREICH
# ===========================================================================

# ---------------------------------------------------------------------------
# SIDEBAR: CSV-IMPORT UND EINSTELLUNGEN
# ---------------------------------------------------------------------------

st.sidebar.header("1. Import")
st.sidebar.caption(f"Version: {VERSION}")

file_type = st.session_state.get('file_type_radio', 'CSV plain')
file_extensions = ["csv"] if file_type == "CSV plain" else ["txt"]

uploaded_file = st.sidebar.file_uploader(
    "upload", type=file_extensions, label_visibility="collapsed",
    help="Datei hochladen. CSV plain: Komma-getrennt. Hubmessung: TAB-getrennt mit fester Samplerate.",
)

# Beim Einklappen des Gesamt-Expanders auch alle Unter-Expander einklappen
_einst_prev = st.session_state.get('_einst_prev', True)
_einst_curr = st.session_state.get('einstellungen', not bool(uploaded_file))
if _einst_prev and not _einst_curr:
    for _k in ('sub_dateityp', 'sub_einlesen', 'sub_kanaele', 'sub_offsets', 'sub_grenzwerte'):
        st.session_state[_k] = False
st.session_state._einst_prev = _einst_curr

with st.sidebar.expander("Einstellungen", expanded=not bool(uploaded_file), key="einstellungen"):

    with st.expander("Dateityp", expanded=st.session_state.sub_dateityp, key="sub_dateityp"):
        st.radio(
            "Dateityp",
            ["CSV plain", "Hubmessung"],
            key="file_type_radio",
            help="CSV plain: Standard-CSV mit Komma-Trennung. Hubmessung: TXT-Datei mit TAB-Trennung und fester Samplerate.",
            on_change=update_sample_rate_for_file_type,
        )

    with st.expander("Einlesen", expanded=st.session_state.sub_einlesen, key="sub_einlesen"):
        sample_rate_unit  = st.session_state.sample_rate_unit
        sample_rate_input = st.number_input(
            "Abtastung",
            min_value=0.0001,
            format="%.3f" if sample_rate_unit == "µs" else "%.1f",
            key="sample_rate",
            help="Hz = Abtastfrequenz, µs = Zeit pro Sample",
        )
        use_us = st.toggle(
            "Hz / µs",
            key="sample_rate_unit_toggle",
            on_change=update_sample_rate_unit,
            label_visibility="visible",
            help="Eingabeeinheit umschalten: µs = Zeitabstand pro Sample, Hz = Abtastfrequenz.",
        )
        sample_rate_unit = "µs" if use_us else "Hz"
        if st.session_state.sample_rate_unit != sample_rate_unit:
            st.session_state.sample_rate_unit = sample_rate_unit
        sample_rate = 1_000_000.0 / sample_rate_input if sample_rate_unit == "µs" else sample_rate_input

        st.number_input("Kopfzeilen überspringen", min_value=0, step=1, key="skip_rows",
                        help="Anzahl der Zeilen am Dateianfang die ignoriert werden (z. B. Metadaten-Header).")
        st.number_input("Max. Samples importieren", min_value=0, step=1000, key="max_samples",
                        help="Maximale Anzahl der zu importierenden Datenpunkte (0 = alle importieren).")

    with st.expander("Kanäle", expanded=st.session_state.sub_kanaele, key="sub_kanaele"):
        st.caption("Leeres Feld = Kanal nicht einlesen")
        st.text_input("Kanal 1 Name", key="ch1_name",
                      help="Leer lassen um Kanal 1 nicht einzulesen.")
        st.text_input("Kanal 2 Name", key="ch2_name",
                      help="Leer lassen um Kanal 2 nicht einzulesen.")
        st.text_input("Kanal 3 Name", key="ch3_name",
                      help="Leer lassen um Kanal 3 nicht einzulesen.")
        st.text_input("Kanal 4 Name", key="ch4_name",
                      help="Leer lassen um Kanal 4 nicht einzulesen.")

    if uploaded_file:
        _kanal_cfg = [
            st.session_state.ch1_name.strip(),
            st.session_state.ch2_name.strip(),
            st.session_state.ch3_name.strip(),
            st.session_state.ch4_name.strip(),
        ]
        kanal_namen_tuple = tuple(n for n in _kanal_cfg if n)

        if len(kanal_namen_tuple) < 1:
            st.sidebar.error("Mindestens ein Kanalname muss angegeben werden.")
            st.stop()

        sensor_namen = list(kanal_namen_tuple)
        file_bytes = uploaded_file.getvalue()

        try:
            df_raw = load_data(
                file_bytes, st.session_state.skip_rows,
                st.session_state.max_samples, kanal_namen_tuple, file_type,
            )
        except ValueError as e:
            st.error(f"Fehler beim Laden: {e}")
            st.stop()

        if st.session_state.last_file_name != uploaded_file.name:
            total_time_ms = len(df_raw) / sample_rate * 1000.0
            for i, name in enumerate(sensor_namen, 1):
                off_init = float(df_raw[name].min()) * -1.0
                st.session_state[f'off{i}']        = off_init
                st.session_state[f'off{i}_slider'] = off_init
            for i in range(len(sensor_namen) + 1, 5):
                st.session_state[f'off{i}']        = 0.0
                st.session_state[f'off{i}_slider'] = 0.0
            st.session_state.xa             = total_time_ms * 0.30
            st.session_state.xb             = total_time_ms * 0.50
            st.session_state.crop_start     = None
            st.session_state.crop_end       = None
            st.session_state.zoom_token    += 1
            st.session_state.last_file_name = uploaded_file.name
            st.rerun()

        with st.expander("Manuelle Offsets (Y)", expanded=st.session_state.sub_offsets, key="sub_offsets"):
            with st.container(border=True):
                st.subheader("Set to 0")
                n_ch     = len(sensor_namen)
                btn_cols = st.columns(n_ch)
                for i, name in enumerate(sensor_namen):
                    if btn_cols[i].button(f"{name}", use_container_width=True,
                                          help="Setzt den Offset so, dass der Minimalwert auf 0 µm liegt.", key=f"auto0_{i}"):
                        val = float(df_raw[name].min()) * -1.0
                        st.session_state[f'off{i+1}']        = val
                        st.session_state[f'off{i+1}_slider'] = val
                        st.rerun()

            st.markdown("")
            for i, name in enumerate(sensor_namen):
                st.slider(
                    f"Offset {name}", -600.0, 600.0, step=0.1,
                    key=f'off{i+1}_slider', on_change=OFF_CALLBACKS[i],
                    help="Y-Versatz für diesen Kanal in µm (Bereich ±600 µm).",
                )

        offs = tuple(st.session_state[f'off{i+1}'] for i in range(len(sensor_namen)))
    else:
        df_raw = None
        sensor_namen = []
        offs = tuple()

    with st.expander("Diagramm-Grenzwerte", expanded=st.session_state.sub_grenzwerte, key="sub_grenzwerte"):
        st.number_input(
            "Geschwindigkeit ± (mm/s)",
            min_value=100,
            max_value=20_000,
            step=100,
            key="v_axis_limit",
        )
        st.number_input(
            "Beschleunigung ± (m/s²)",
            min_value=1_000,
            max_value=50_000,
            step=500,
            key="a_axis_limit",
        )

if sample_rate <= 0:
    st.sidebar.error("Samplerate muss größer als 0 sein.")
    st.stop()

if not uploaded_file:
    st.info("Bitte laden Sie eine CSV-Datei hoch, um die Analyse zu starten.")
    st.stop()

# ---------------------------------------------------------------------------
# KANAL-KONFIGURATION – aktive Kanäle aus Einstellungen ableiten
# ---------------------------------------------------------------------------

_kanal_cfg = [
    st.session_state.ch1_name.strip(),
    st.session_state.ch2_name.strip(),
    st.session_state.ch3_name.strip(),
    st.session_state.ch4_name.strip(),
]
kanal_namen_tuple = tuple(n for n in _kanal_cfg if n)

if len(kanal_namen_tuple) < 1:
    st.sidebar.error("Mindestens ein Kanalname muss angegeben sein.")
    st.stop()

# ---------------------------------------------------------------------------
# DATEN LADEN
# ---------------------------------------------------------------------------

file_bytes = uploaded_file.getvalue()

try:
    df_raw = load_data(
        file_bytes, st.session_state.skip_rows,
        st.session_state.max_samples, kanal_namen_tuple, file_type,
    )
except ValueError as e:
    st.error(f"Fehler beim Laden: {e}")
    st.stop()

sensor_namen = list(kanal_namen_tuple)   # tatsächlich geladene Kanalnamen

# ---------------------------------------------------------------------------
# AUTO-RESET BEI NEUER DATEI
# ---------------------------------------------------------------------------

if st.session_state.last_file_name != uploaded_file.name:
    total_time_ms = len(df_raw) / sample_rate * 1000.0
    for i, name in enumerate(sensor_namen, 1):
        off_init = float(df_raw[name].min()) * -1.0
        st.session_state[f'off{i}']        = off_init
        st.session_state[f'off{i}_slider'] = off_init
    # Offsets für nicht geladene Kanäle zurücksetzen
    for i in range(len(sensor_namen) + 1, 5):
        st.session_state[f'off{i}']        = 0.0
        st.session_state[f'off{i}_slider'] = 0.0
    st.session_state.xa             = total_time_ms * 0.30
    st.session_state.xb             = total_time_ms * 0.50
    st.session_state.crop_start     = None
    st.session_state.crop_end       = None
    st.session_state.zoom_token    += 1
    st.session_state.last_file_name = uploaded_file.name
    st.rerun()

# Offsets für alle aktiven Kanäle auslesen
offs = tuple(st.session_state[f'off{i+1}'] for i in range(len(sensor_namen)))

# ---------------------------------------------------------------------------
# DATENVERARBEITUNG
# ---------------------------------------------------------------------------

if file_type == "Hubmessung":
    # Für Hubmessungen: Zeit aus der Datei verwenden
    zeit_full = df_raw['Zeit (ms)'].values
else:
    # Für CSV: Zeitachse berechnen
    zeit_full = build_time_axis(len(df_raw), sample_rate)   # ms

max_zeit_full = float(zeit_full[-1])
max_idx_full  = len(df_raw) - 1

kanal_arrays = tuple(df_raw[name].values for name in sensor_namen)
df_full = apply_offsets(kanal_namen_tuple, kanal_arrays, offs, zeit_full)

# ---------------------------------------------------------------------------
# CROP-LOGIK
# ---------------------------------------------------------------------------

crop_active = (
    st.session_state.crop_start is not None
    and st.session_state.crop_end is not None
)
if crop_active:
    ci_start = get_idx_at_x(st.session_state.crop_start, sample_rate, max_idx_full)
    ci_end   = get_idx_at_x(st.session_state.crop_end,   sample_rate, max_idx_full)
    df       = df_full.iloc[ci_start:ci_end + 1].reset_index(drop=True)
    min_zeit = float(df['Zeit (ms)'].iloc[0])
    max_zeit = float(df['Zeit (ms)'].iloc[-1])
    max_idx  = len(df) - 1
else:
    df       = df_full
    min_zeit = 0.0
    max_zeit = max_zeit_full
    max_idx  = max_idx_full

# ---------------------------------------------------------------------------
# SIDEBAR: AUSWERTUNGS-STEUERUNG
# ---------------------------------------------------------------------------

st.sidebar.header("2. Auswertung")
active_sensor = st.sidebar.radio(
    "Kanal für Messung:", sensor_namen,
    horizontal=True, label_visibility="collapsed",
    help="Aktiver Kanal für alle Berechnungen: Cursor-Messung, v-max, a-max und SOP.",
)

# Cursor-Werte auf aktiven Zeitbereich begrenzen
xa = float(np.clip(st.session_state.xa, min_zeit, max_zeit))
xb = float(np.clip(st.session_state.xb, min_zeit, max_zeit))
st.session_state.xa = xa
st.session_state.xb = xb

with st.sidebar.expander("Zeitmarker & Basis", expanded=False):
    st.number_input(
        "Zeit XA (ms)", min_zeit, max_zeit,
        value=xa, step=0.001, format="%.3f",
        key="xa_nw", on_change=update_xa_from_num,
        help="Linker Zeitcursor in ms – Startpunkt für Δt, Δs und v-mid.",
    )
    st.number_input(
        "Zeit XB (ms)", min_zeit, max_zeit,
        value=xb, step=0.001, format="%.3f",
        key="xb_nw", on_change=update_xb_from_num,
        help="Rechter Zeitcursor in ms – Endpunkt für Δt, Δs und v-mid.",
    )
    if xa > xb:
        st.warning("⚠️ XA liegt nach XB – Marker vertauscht.")
    v_time_base_ms = st.slider(
        "Zeitbasis v-max (ms)", 0.005, 0.10, 0.03,
        step=0.005, format="%.3f ms",
        help="Mittelungsfenster für v-max, a-max und SOP: Der Peak wird über dieses Zeitfenster gemittelt. Kleiner = empfindlicher, größer = robuster gegenüber Rauschen.",
    )

show_v_avg    = st.sidebar.toggle("v-Schnitt Linie (A-B) anzeigen", value=False,
                                  help="Zeichnet eine Verbindungslinie von XA nach XB und visualisiert damit die mittlere Geschwindigkeit v-mid.")
show_rect_fit = st.sidebar.toggle(
    "Best-fit Rechteck füllen", value=False,
    help="Zeigt zusätzlich vertikale Kantenlinien und hellgrüne Füllung für alle erkannten Rechteck-Pulse.",
)
show_velocity = st.sidebar.toggle(
    "Geschwindigkeit anzeigen", value=False,
    help="Zeigt die Geschwindigkeit (mm/s) des aktiven Kanals auf einer zweiten Y-Achse rechts. Achse fest auf ±3200 mm/s.",
)
if show_velocity:
    st.sidebar.slider(
        "Glättung Geschwindigkeit", 5, 80, step=1,
        value=st.session_state.window_length,
        key="window_length",
        help="Fenstergröße des Savitzky-Golay-Filters für die Geschwindigkeitskurve. Größer = glatter, aber geringere Detailauflösung.",
    )
show_acceleration = st.sidebar.toggle(
    "Beschleunigung anzeigen", value=False,
    help="Zeigt die Beschleunigung (m/s²) des aktiven Kanals auf einer dritten Y-Achse rechts. Achse fest auf ±12000 m/s².",
)
if show_acceleration:
    st.sidebar.slider(
        "Glättung Beschleunigung", 10, 75, step=1,
        value=st.session_state.window_length_accel,
        key="window_length_accel",
        help="Fenstergröße des Savitzky-Golay-Filters für die Beschleunigungskurve. Größere Werte nötig, da die 2. Ableitung stärker rauscht.",
    )

show_sop = st.sidebar.toggle(
    "Speed on Point (SOP)", value=False,
    help="Misst die Geschwindigkeit an der steigenden Flanke des Rechtecksignals auf einem einstellbaren Hub-Pegel. Erfordert erkanntes Rechteck-Fit.",
)
if show_sop:
    st.sidebar.slider(
        "SOP Pegel (%)", 0, 100, step=1,
        value=st.session_state.sop_percent,
        key="sop_percent",
        help="Höhe auf der steigenden Flanke in Prozent des Hub (0 % = unterer Pegel, 100 % = oberer Pegel).",
    )

# Rechteck-Fit auf den vollständigen (ungecropten) Datensatz anwenden
rect_fit = compute_best_fit_rectangle(
    df_full['Zeit (ms)'].values,
    df_full[active_sensor].values,
)

# ---------------------------------------------------------------------------
# MESSWERTBERECHNUNG
# ---------------------------------------------------------------------------

# Indizes der Cursor-Positionen im (ggf. gecropten) DataFrame
if crop_active:
    idx_a = get_idx_at_x(xa - min_zeit, sample_rate, max_idx)
    idx_b = get_idx_at_x(xb - min_zeit, sample_rate, max_idx)
else:
    idx_a = get_idx_at_x(xa, sample_rate, max_idx)
    idx_b = get_idx_at_x(xb, sample_rate, max_idx)

ya = df.loc[idx_a, active_sensor]
yb = df.loc[idx_b, active_sensor]

dt_val_ms = abs(xb - xa)                                   # ms
dy_um     = abs(yb - ya)                                   # µm
v_avg     = dy_um / dt_val_ms if dt_val_ms > 0 else 0.0   # mm/s  (µm/ms = mm/s)

# Momentangeschwindigkeit an XA und XB über ein Zeitbasis-Fenster
halbes_zeitfenster = max(1, int(v_time_base_ms / 1000.0 * sample_rate / 2))

def v_at_cursor(idx: int) -> float:
    """Mittlere Momentangeschwindigkeit um idx herum (mm/s)."""
    i0 = max(0, idx - halbes_zeitfenster)
    i1 = min(max_idx, idx + halbes_zeitfenster)
    if i1 <= i0:
        return float('nan')
    dy   = float(df.loc[i1, active_sensor] - df.loc[i0, active_sensor])
    dt_s = (i1 - i0) / sample_rate
    return (dy / 1000.0) / dt_s   # µm → mm, s → mm/s

v_at_xa        = v_at_cursor(idx_a)
v_at_xb        = v_at_cursor(idx_b)
v_cursor_delta = (
    abs(v_at_xb - v_at_xa)
    if not (np.isnan(v_at_xa) or np.isnan(v_at_xb))
    else float('nan')
)

idx_start, idx_end = sorted([idx_a, idx_b])

# Initialisierung der Peak-Marker (werden nur gesetzt wenn genug Datenpunkte vorhanden)
t_vmax_start, y_vmax_start = None, None
t_vmax_ende,  y_vmax_ende  = None, None
t_amax_falling, y_amax_falling = None, None
t_amax_rising,  y_amax_rising  = None, None
has_vmax         = False
has_amax_falling = False
has_amax_rising  = False
v_max            = float('nan')
a_max_falling    = float('nan')
a_min_rising     = float('nan')
sop_linien: list = []
v_sop            = float('nan')

if idx_end > idx_start:
    arr_full  = df[active_sensor].values
    dt_step_s = 1.0 / sample_rate

    # v-max: SG-Filter auf dem vollständigen Datensatz – verhindert Randeffekte
    gefilt_geschw_roh_full = _berechne_sg_ableitung(arr_full, dt_step_s, st.session_state.window_length, 1)
    if gefilt_geschw_roh_full is not None:
        abs_geschw_full = np.abs(gefilt_geschw_roh_full / 1000.0)   # µm/s → mm/s
        # Peak nur im Cursor-Bereich suchen
        abs_geschw_slice  = abs_geschw_full[idx_start:idx_end + 1]
        idx_vmax_peak_loc = int(np.argmax(abs_geschw_slice))
        idx_vmax_peak     = idx_start + idx_vmax_peak_loc
        iv_start          = max(0, idx_vmax_peak - halbes_zeitfenster)
        iv_ende           = min(max_idx, idx_vmax_peak + halbes_zeitfenster)
        v_max             = min(float(np.mean(abs_geschw_full[iv_start:iv_ende + 1])),
                                float(st.session_state.v_axis_limit))

        if 0 <= iv_start <= max_idx and 0 <= iv_ende <= max_idx:
            t_vmax_start = df.loc[iv_start, 'Zeit (ms)']
            y_vmax_start = df.loc[iv_start, active_sensor]
            t_vmax_ende  = df.loc[iv_ende,  'Zeit (ms)']
            y_vmax_ende  = df.loc[iv_ende,  active_sensor]
            has_vmax     = True

    # a-max: SG-Filter auf dem vollständigen Datensatz – verhindert Randeffekte an Cursor-Grenzen
    gefilt_beschl_roh_full = _berechne_sg_ableitung(arr_full, dt_step_s, st.session_state.window_length_accel, 2)
    if gefilt_beschl_roh_full is not None:
        gefilt_beschl_full = gefilt_beschl_roh_full / 1_000_000.0   # µm/s² → m/s²

        def _peak_marker(idx_abs):
            """Gemittelter Beschleunigungswert und Diagramm-Position für einen Peak (absoluter Index)."""
            ia0  = max(0, idx_abs - halbes_zeitfenster)
            ia1  = min(max_idx, idx_abs + halbes_zeitfenster)
            wert = float(np.mean(gefilt_beschl_full[ia0:ia1 + 1]))
            return wert, float(df.loc[idx_abs, 'Zeit (ms)']), float(df.loc[idx_abs, active_sensor])

        # Peak nur im Cursor-Bereich suchen
        beschl_slice = gefilt_beschl_full[idx_start:idx_end + 1]

        a_lim = float(st.session_state.a_axis_limit)

        idx_falling_abs                                = idx_start + int(np.argmax(beschl_slice))
        a_max_falling, t_amax_falling, y_amax_falling = _peak_marker(idx_falling_abs)
        a_max_falling = min(a_max_falling,  a_lim)
        has_amax_falling = True

        idx_rising_abs                               = idx_start + int(np.argmin(beschl_slice))
        a_min_rising, t_amax_rising, y_amax_rising   = _peak_marker(idx_rising_abs)
        a_min_rising = max(a_min_rising, -a_lim)
        has_amax_rising = True

# SOP – steht nach halbes_zeitfenster-Definition und nach rect_fit
if show_sop and rect_fit is not None:
    sop_linien, v_sop = _finde_sop_kreuzungen(
        df_full['Zeit (ms)'].values,
        df_full[active_sensor].values,
        rect_fit,
        st.session_state.sop_percent,
        sample_rate,
        halbes_zeitfenster,
    )

# ---------------------------------------------------------------------------
# DOWNSAMPLING FÜR GROSSE DATEIEN
# ---------------------------------------------------------------------------

if len(df) > MAX_PLOT_PUNKTE:
    step    = len(df) // MAX_PLOT_PUNKTE
    df_plot = df.iloc[::step]
else:
    df_plot = df

# ---------------------------------------------------------------------------
# ABLEITUNGEN FÜR DIAGRAMM-DARSTELLUNG
# ---------------------------------------------------------------------------

velocity, acceleration = _berechne_ableitungen_fuer_diagramm(
    df_plot, active_sensor, show_velocity, show_acceleration
)

# ---------------------------------------------------------------------------
# DIAGRAMM AUFBAUEN
# ---------------------------------------------------------------------------

fig = go.Figure()

for i, name in enumerate(sensor_namen):
    fig.add_trace(go.Scatter(
        x=df_plot['Zeit (ms)'], y=df_plot[name],
        name=name, line=dict(color=KANAL_FARBEN[i]),
    ))

fig.add_vline(x=xa, line_dash="dash", line_color=FARBE_CURSOR)
fig.add_vline(x=xb, line_dash="dash", line_color=FARBE_CURSOR)

if show_v_avg:
    fig.add_trace(go.Scatter(
        x=[xa, xb], y=[ya, yb],
        mode='lines+markers', name='v-Schnitt',
        line=dict(color=FARBE_V_SCHNITT, width=2, dash='dot'),
    ))

if rect_fit is not None:
    _zeichne_rechteck_fit(fig, rect_fit, min_zeit, max_zeit, mit_fuellung=show_rect_fit)

if has_vmax:
    fig.add_trace(go.Scatter(
        x=[t_vmax_start, t_vmax_ende], y=[y_vmax_start, y_vmax_ende],
        mode='lines+markers', name='v-max',
        line=dict(color=FARBE_VMAX, width=2),
    ))
if has_amax_falling:
    fig.add_trace(go.Scatter(
        x=[t_amax_falling], y=[y_amax_falling],
        mode='markers', name='a-max',
        marker=dict(color=FARBE_AMAX, size=14, symbol='cross',
                    line=dict(color=FARBE_AMAX, width=2)),
    ))
if has_amax_rising:
    fig.add_trace(go.Scatter(
        x=[t_amax_rising], y=[y_amax_rising],
        mode='markers', name='a-min',
        marker=dict(color=FARBE_AMAX, size=12, symbol='circle',
                    line=dict(color=FARBE_AMAX, width=2)),
    ))
if sop_linien:
    erste_sichtbar = True
    for t_sop, t0, t1, y_lvl in sop_linien:
        if not (min_zeit <= t_sop <= max_zeit):
            continue
        fig.add_trace(go.Scatter(
            x=[max(t0, min_zeit), min(t1, max_zeit)], y=[y_lvl, y_lvl],
            mode='lines',
            name='SOP' if erste_sichtbar else None,
            showlegend=erste_sichtbar,
            line=dict(color=FARBE_GESCHW, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=[t_sop], y=[y_lvl],
            mode='markers', showlegend=False,
            marker=dict(color=FARBE_GESCHW, size=14, symbol='x',
                        line=dict(color=FARBE_GESCHW, width=2)),
        ))
        erste_sichtbar = False
if show_velocity and velocity is not None:
    fig.add_trace(go.Scatter(
        x=df_plot['Zeit (ms)'], y=velocity,
        name='Geschwindigkeit', yaxis='y2', line=dict(color=FARBE_GESCHW),
    ))
if show_acceleration and acceleration is not None:
    fig.add_trace(go.Scatter(
        x=df_plot['Zeit (ms)'], y=acceleration,
        name='Beschleunigung', yaxis='y3', line=dict(color=FARBE_BESCHL),
    ))

# Y-Achse: 15 % Puffer über Signalbereich damit Legende den Graph nicht verdeckt
y_max_plot   = float(df_plot[sensor_namen].max().max())
y_min_plot   = float(df_plot[sensor_namen].min().min())
y_range_plot = [y_min_plot, y_max_plot + (y_max_plot - y_min_plot) * 0.15]

fig.update_layout(
    xaxis_title="Zeit (ms)",
    yaxis_title="Weg (µm)",
    height=600,
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
    uirevision=f"{st.session_state.zoom_token}-{st.session_state.crop_start}-{st.session_state.crop_end}",
    xaxis=dict(autorange=True, rangemode='nonnegative'),
    yaxis=dict(range=y_range_plot),
)
if show_velocity and velocity is not None:
    fig.update_layout(
        yaxis2=dict(
            title='Geschwindigkeit (mm/s)',
            overlaying='y', side='right', showgrid=False,
            range=[-st.session_state.v_axis_limit, st.session_state.v_axis_limit],
        )
    )
if show_acceleration and acceleration is not None:
    fig.update_layout(
        yaxis3=dict(
            title='Beschleunigung (m/s²)',
            overlaying='y', side='right', showgrid=False,
            position=0.85 if show_velocity else 1.0,
            range=[-st.session_state.a_axis_limit, st.session_state.a_axis_limit],
        )
    )
st.plotly_chart(fig, width="stretch", key="main_chart")

# ---------------------------------------------------------------------------
# CURSOR-SLIDER
# 0.04-Spalte gleicht Y-Achsen-Breite aus damit Slider mit Diagramm fluchten
# ---------------------------------------------------------------------------

c_pad, c_slider = st.columns([0.04, 0.96])
with c_slider:
    st.slider(
        "XA", min_zeit, max_zeit, value=xa,
        key="xa_sw", step=0.001, format="%.3f ms",
        on_change=update_xa_from_slider, label_visibility="collapsed",
        help="Linker Cursor XA (ms) – ziehen oder Wert im Expander 'Zeitmarker & Basis' eingeben.",
    )
    st.slider(
        "XB", min_zeit, max_zeit, value=xb,
        key="xb_sw", step=0.001, format="%.3f ms",
        on_change=update_xb_from_slider, label_visibility="collapsed",
        help="Rechter Cursor XB (ms) – ziehen oder Wert im Expander 'Zeitmarker & Basis' eingeben.",
    )

# ---------------------------------------------------------------------------
# CROP / SHOW ALL
# ---------------------------------------------------------------------------

margin  = abs(xb - xa) * 0.15
crop_t0 = max(min_zeit, min(xa, xb) - margin)
crop_t1 = min(max_zeit, max(xa, xb) + margin)

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    if st.button("✂️ Crop A–B  (+15%)", disabled=(dt_val_ms == 0), width="stretch",
                 help="Schneidet die Ansicht auf den Bereich zwischen XA und XB zu (je 15 % Rand beiderseits)."):
        st.session_state.crop_start  = crop_t0
        st.session_state.crop_end    = crop_t1
        st.session_state.xa          = float(min(xa, xb))
        st.session_state.xb          = float(max(xa, xb))
        st.session_state.zoom_token += 1
        st.rerun()
with btn_col2:
    if st.button("🔍 Show All", disabled=not crop_active, width="stretch",
                 help="Setzt den Crop zurück und zeigt den gesamten Messzeitraum."):
        st.session_state.crop_start  = None
        st.session_state.crop_end    = None
        st.session_state.zoom_token += 1
        st.rerun()

if crop_active:
    st.caption(
        f"✂️ Crop aktiv: {st.session_state.crop_start:.3f} ms – "
        f"{st.session_state.crop_end:.3f} ms"
    )

# ---------------------------------------------------------------------------
# KENNGRÖSSEN-ANZEIGE
# ---------------------------------------------------------------------------

freq_hz = (1000.0 / dt_val_ms) if dt_val_ms > 0 else float('nan')
hub_um  = abs(rect_fit['y_high'] - rect_fit['y_low']) if rect_fit is not None else float('nan')

# Zeile 1 – Zeit & Weg
z1, z2, z3, z4 = st.columns(4)
z1.metric("Δt (A-B)",          f"{dt_val_ms:.3f} ms")
z2.metric("Frequenz Δt (A-B)", f"{freq_hz:.1f} Hz"          if not np.isnan(freq_hz) else "N/A")
z3.metric("Δs (A-B)",          f"{dy_um:.1f} µm")
z4.metric("Hub Best-fit",      f"{hub_um:.1f} µm"           if not np.isnan(hub_um) else "N/A")

# Zeile 2 – Geschwindigkeit (alle mm/s)
g1, g2, g3, g4 = st.columns(4)
g1.metric("v-mid (A-B)",       f"{v_avg:.1f} mm/s")
g2.metric("Δv Cursor (A-B)",   f"{v_cursor_delta:.1f} mm/s" if not np.isnan(v_cursor_delta) else "N/A")
g3.metric("v-max (Peak)",      f"{v_max:.1f} mm/s"          if not np.isnan(v_max) else "N/A")
g4.metric("SOP",               f"{v_sop:.1f} mm/s"          if not np.isnan(v_sop) else "N/A")

# Zeile 3 – Beschleunigung (beide m/s²)
a1, a2 = st.columns(2)
a1.metric("a-max Falling",     f"{a_max_falling:.0f} m/s²"  if not np.isnan(a_max_falling) else "N/A")
a2.metric("a-min Rising",      f"{a_min_rising:.0f} m/s²"   if not np.isnan(a_min_rising) else "N/A")

# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------

st.sidebar.header("3. Export")
metrics = {
    # Zeit & Weg
    "XA (ms)":              f"{xa:.3f}",
    "XB (ms)":              f"{xb:.3f}",
    "Δt (A-B)":             f"{dt_val_ms:.3f} ms",
    "Frequenz Δt (A-B)":    f"{freq_hz:.1f} Hz"           if not np.isnan(freq_hz) else "N/A",
    "Δs (A-B)":             f"{dy_um:.1f} µm",
    "Hub Best-fit":         f"{hub_um:.1f} µm"            if not np.isnan(hub_um) else "N/A",
    # Geschwindigkeit
    "v-mid (A-B)":          f"{v_avg:.1f} mm/s",
    "Δv Cursor (A-B)":      f"{v_cursor_delta:.1f} mm/s"  if not np.isnan(v_cursor_delta) else "N/A",
    "v-max (Peak)":         f"{v_max:.1f} mm/s"           if not np.isnan(v_max) else "N/A",
    "SOP":                  f"{v_sop:.1f} mm/s"           if not np.isnan(v_sop) else "N/A",
    # Beschleunigung
    "a-max Falling":        f"{a_max_falling:.0f} m/s²"   if not np.isnan(a_max_falling) else "N/A",
    "a-min Rising":         f"{a_min_rising:.0f} m/s²"    if not np.isnan(a_min_rising) else "N/A",
}
export_format = st.sidebar.radio(
    "Format:", ["PDF", "PNG"], horizontal=True, label_visibility="collapsed",
    help="PDF enthält Diagramm und Kenngrößen-Tabelle; PNG ist nur das Diagramm.",
)
if st.sidebar.button("📥 Export erstellen", width="stretch",
                     help="Erstellt die Exportdatei im gewählten Format – Download-Button erscheint danach."):
    with st.spinner("Wird erstellt..."):
        try:
            chart_png = build_chart_png(
                df, sensor_namen, active_sensor,
                xa, xb, ya, yb, show_v_avg,
                t_vmax_start, y_vmax_start, t_vmax_ende, y_vmax_ende, has_vmax,
                t_amax_falling, y_amax_falling, has_amax_falling,
                t_amax_rising,  y_amax_rising,  has_amax_rising,
                show_rect_fit=show_rect_fit,
                rect_fit=rect_fit,
                show_velocity=show_velocity,
                window_length=st.session_state.window_length,
                show_acceleration=show_acceleration,
                window_length_accel=st.session_state.window_length_accel,
                sop_linien=sop_linien,
            )
            stem = uploaded_file.name.rsplit('.', 1)[0]
            if export_format == "PDF":
                file_bytes_out = build_pdf(uploaded_file.name, chart_png, metrics)
                st.sidebar.download_button(
                    label="💾 PDF herunterladen",
                    data=file_bytes_out,
                    file_name=f"{stem}_auswertung.pdf",
                    mime="application/pdf",
                    width="stretch",
                )
            else:
                st.sidebar.download_button(
                    label="💾 PNG herunterladen",
                    data=chart_png,
                    file_name=f"{stem}_diagramm.png",
                    mime="image/png",
                    width="stretch",
                )
        except Exception as exc:
            st.sidebar.error(f"Export fehlgeschlagen: {exc}")
