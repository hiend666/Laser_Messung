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

VERSION = "v1.00.05"

# ---------------------------------------------------------------------------
# KONSTANTEN
# ---------------------------------------------------------------------------

V_ACHSE_LIMIT_MM_S = 3_200    # Feste Y-Grenze Geschwindigkeitsachse  ± mm/s
A_ACHSE_LIMIT_M_S2 = 12_000   # Feste Y-Grenze Beschleunigungsachse   ± m/s²

MAX_PLOT_PUNKTE    = 5_000     # Downsampling-Schwelle für interaktives Diagramm
SAVGOL_POLYNOM     = 3         # Polynomgrad für alle Savitzky-Golay-Filter

# Diagramm-Farben
FARBE_KANAL1    = '#003366'
FARBE_KANAL2    = '#4c78a8'
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
    'off1_slider': 0.0,
    'off2_slider': 0.0,
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
    'max_samples': 8000,
    # Crop-State: None = "Show All", sonst t_start / t_end als float
    'crop_start': None,
    'crop_end': None,
    'show_velocity': False,
    'window_length': 50,
    'show_acceleration': False,
    'window_length_accel': 75,
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
    return savgol_filter(signal, fenster, SAVGOL_POLYNOM, deriv=ordnung, delta=dt_s)


# ---------------------------------------------------------------------------
# GECACHTE DATENFUNKTIONEN
# ---------------------------------------------------------------------------

@st.cache_data
def load_csv(
    file_bytes: bytes, skip_rows: int, max_samples: int, ch1: str, ch2: str
) -> pd.DataFrame:
    """Liest CSV im Roh- oder Sauber-Format und gibt DataFrame mit zwei Messspalten zurück."""
    nrows = max_samples if max_samples > 0 else None
    probe = pd.read_csv(
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
        data_cols = [c for c in df.columns if df[c].dtype in ['float64', 'float32']]
        if len(data_cols) < 2:
            raise ValueError("CSV enthält weniger als 2 numerische Spalten.")
        df = df[data_cols[:2]].copy()
        df.columns = [ch1, ch2]
    except ValueError as exc:
        # Sauber-Format: erste Zeile ist Spaltenheader
        df = pd.read_csv(io.BytesIO(file_bytes), sep=',', decimal='.', nrows=nrows)
        df = df.dropna(axis=1, how='all')
        sensor_cols = [c for c in df.columns if c != 'Zeit (s)']
        if len(sensor_cols) < 2:
            raise ValueError(
                "CSV enthält weniger als 2 Messwert-Spalten nach dem Header."
            ) from exc
        df = df.rename(columns={sensor_cols[0]: ch1, sensor_cols[1]: ch2})
    return df


@st.cache_data
def build_time_axis(n_samples: int, sr: float) -> np.ndarray:
    """Erzeugt Zeitvektor in ms für n_samples bei Samplerate sr (Hz)."""
    return np.arange(n_samples) * (1000.0 / sr)


@st.cache_data
def apply_offsets(
    raw_s1: np.ndarray, raw_s2: np.ndarray,
    off1: float, off2: float,
    zeit: np.ndarray,
    s1: str, s2: str,
) -> pd.DataFrame:
    """Erzeugt den verarbeiteten DataFrame – nur bei echten Änderungen neu berechnet."""
    return pd.DataFrame({'Zeit (ms)': zeit, s1: raw_s1 + off1, s2: raw_s2 + off2})


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
# Widgets schreiben immer in den freien Key (xa/xb/off1/off2), nie umgekehrt.
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

def update_sample_rate_unit():
    new_unit = "µs" if st.session_state.sample_rate_unit_toggle else "Hz"
    old_unit = st.session_state.sample_rate_unit
    if new_unit != old_unit:
        if st.session_state.sample_rate > 0:
            st.session_state.sample_rate = 1_000_000.0 / st.session_state.sample_rate
        st.session_state.sample_rate_unit = new_unit


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
# EXPORT: DIAGRAMM ALS PNG
# ---------------------------------------------------------------------------

def build_chart_png(
    df, s1_name, s2_name, active_sensor,
    xa, xb, ya, yb, show_v_avg,
    t_vmax_start, y_vmax_start, t_vmax_ende, y_vmax_ende, has_vmax,
    t_amax, y_amax, has_amax,
    show_rect_fit=False, rect_fit=None,
    show_velocity=False, window_length=21,
    show_acceleration=False, window_length_accel=21,
) -> bytes:
    """Rendert das Diagramm mit Kaleido zu PNG-Bytes für den Export."""

    # Y-Achse: 15 % Puffer über Signalbereich damit Legende den Graph nicht verdeckt
    y_max_e   = float(df[[s1_name, s2_name]].max().max())
    y_min_e   = float(df[[s1_name, s2_name]].min().min())
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
    export_fig.add_trace(go.Scatter(
        x=df['Zeit (ms)'], y=df[s1_name], name=s1_name, line=dict(color=FARBE_KANAL1),
    ))
    export_fig.add_trace(go.Scatter(
        x=df['Zeit (ms)'], y=df[s2_name], name=s2_name, line=dict(color=FARBE_KANAL2),
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
    if has_amax and t_amax is not None:
        export_fig.add_trace(go.Scatter(
            x=[t_amax], y=[y_amax],
            mode='markers', name='a-max',
            marker=dict(color=FARBE_AMAX, size=14, symbol='cross',
                        line=dict(color=FARBE_AMAX, width=2)),
        ))
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
                range=[-V_ACHSE_LIMIT_MM_S, V_ACHSE_LIMIT_MM_S],
            )
        )
    if show_acceleration and acceleration is not None:
        export_fig.update_layout(
            yaxis3=dict(
                title='Beschleunigung (m/s²)',
                overlaying='y', side='right', showgrid=False,
                position=0.85 if show_velocity else 1.0,
                range=[-A_ACHSE_LIMIT_M_S2, A_ACHSE_LIMIT_M_S2],
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

st.sidebar.header("1. CSV-Import")
st.sidebar.caption(f"Version: {VERSION}")
uploaded_file = st.sidebar.file_uploader("upload", type="csv", label_visibility="collapsed")

with st.sidebar.expander("⚙️ Einstellungen", expanded=not bool(uploaded_file)):
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
    )
    sample_rate_unit = "µs" if use_us else "Hz"
    if st.session_state.sample_rate_unit != sample_rate_unit:
        st.session_state.sample_rate_unit = sample_rate_unit
    sample_rate = 1_000_000.0 / sample_rate_input if sample_rate_unit == "µs" else sample_rate_input

    st.number_input("Kopfzeilen überspringen", min_value=0, step=1, key="skip_rows")
    st.number_input("Max. Samples importieren", min_value=0, step=1000, key="max_samples",
                    help="Maximale Anzahl der zu importierenden Datenpunkte (0 = alle)")
    st.text_input("Kanal 1 Name", key="ch1_name")
    st.text_input("Kanal 2 Name", key="ch2_name")

if sample_rate <= 0:
    st.sidebar.error("Samplerate muss größer als 0 sein.")
    st.stop()

if not uploaded_file:
    st.info("Bitte laden Sie eine CSV-Datei hoch, um die Analyse zu starten.")
    st.stop()

# ---------------------------------------------------------------------------
# DATEN LADEN
# ---------------------------------------------------------------------------

file_bytes = uploaded_file.read()
ch1 = st.session_state.ch1_name or 'Kanal 1'
ch2 = st.session_state.ch2_name or 'Kanal 2'

try:
    df_raw = load_csv(file_bytes, st.session_state.skip_rows, st.session_state.max_samples, ch1, ch2)
except ValueError as e:
    st.error(f"Fehler beim Laden: {e}")
    st.stop()

sensor_cols = [c for c in df_raw.columns if c != 'Zeit (s)']
if len(sensor_cols) < 2:
    st.error("Die Datei benötigt mindestens zwei Messwert-Spalten.")
    st.stop()

s1_name, s2_name = sensor_cols[0], sensor_cols[1]

# ---------------------------------------------------------------------------
# AUTO-RESET BEI NEUER DATEI
# ---------------------------------------------------------------------------

if st.session_state.last_file_name != uploaded_file.name:
    total_time_ms = len(df_raw) / sample_rate * 1000.0
    off1_init = float(df_raw[s1_name].min()) * -1.0
    off2_init = float(df_raw[s2_name].min()) * -1.0
    st.session_state.off1           = off1_init
    st.session_state.off2           = off2_init
    st.session_state.off1_slider    = off1_init
    st.session_state.off2_slider    = off2_init
    st.session_state.xa             = total_time_ms * 0.30
    st.session_state.xb             = total_time_ms * 0.50
    st.session_state.crop_start     = None
    st.session_state.crop_end       = None
    st.session_state.zoom_token    += 1
    st.session_state.last_file_name = uploaded_file.name
    st.rerun()

# ---------------------------------------------------------------------------
# SIDEBAR: MANUELLE Y-OFFSETS
# ---------------------------------------------------------------------------

with st.sidebar.expander("⚖️ Manuelle Offsets (Y)", expanded=False):
    col1, col2 = st.columns(2)
    if col1.button(f"Auto-0\n{s1_name}", use_container_width=True):
        val = float(df_raw[s1_name].min()) * -1.0
        st.session_state.off1        = val
        st.session_state.off1_slider = val
        st.rerun()
    if col2.button(f"Auto-0\n{s2_name}", use_container_width=True):
        val = float(df_raw[s2_name].min()) * -1.0
        st.session_state.off2        = val
        st.session_state.off2_slider = val
        st.rerun()
    st.slider(
        f"Offset {s1_name}", -600.0, 600.0, step=0.1,
        key="off1_slider", on_change=update_off1_from_slider,
    )
    st.slider(
        f"Offset {s2_name}", -600.0, 600.0, step=0.1,
        key="off2_slider", on_change=update_off2_from_slider,
    )

off1 = st.session_state.off1
off2 = st.session_state.off2

# ---------------------------------------------------------------------------
# DATENVERARBEITUNG
# ---------------------------------------------------------------------------

zeit_full     = build_time_axis(len(df_raw), sample_rate)   # ms
max_zeit_full = float(zeit_full[-1])
max_idx_full  = len(df_raw) - 1

df_full = apply_offsets(
    df_raw[s1_name].values, df_raw[s2_name].values,
    off1, off2, zeit_full, s1_name, s2_name,
)

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
    "Kanal für Messung:", [s1_name, s2_name],
    horizontal=True, label_visibility="collapsed",
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
    )
    st.number_input(
        "Zeit XB (ms)", min_zeit, max_zeit,
        value=xb, step=0.001, format="%.3f",
        key="xb_nw", on_change=update_xb_from_num,
    )
    if xa > xb:
        st.warning("⚠️ XA liegt nach XB – Marker vertauscht.")
    # Zeitbasis für die Mittelwertbildung bei v-max und a-max (in ms)
    v_time_base_ms = st.slider(
        "Zeitbasis v-max (ms)", 0.01, 0.10, 0.05,
        step=0.01, format="%.2f ms",
    )

show_v_avg    = st.sidebar.toggle("v-Schnitt Linie (A-B) anzeigen", value=False)
show_rect_fit = st.sidebar.toggle(
    "Best-fit Rechteck füllen", value=False,
    help="Zeigt vertikale Kanten und hellgrüne Füllung für alle erkannten Pulse.",
)
show_velocity = st.sidebar.toggle(
    "Geschwindigkeit anzeigen", value=False,
    help="Zeigt die Geschwindigkeit des ausgewählten Kanals mit zweiter Y-Achse rechts.",
)
if show_velocity:
    st.sidebar.slider(
        "Glättung Geschwindigkeit", 10, 90, step=1,
        value=st.session_state.window_length,
        key="window_length",
        help="Fenstergröße Savitzky-Golay-Filter (größer = glatter, aber weniger Details).",
    )
show_acceleration = st.sidebar.toggle(
    "Beschleunigung anzeigen", value=False,
    help="Zeigt die Beschleunigung des ausgewählten Kanals mit dritter Y-Achse.",
)
if show_acceleration:
    st.sidebar.slider(
        "Glättung Beschleunigung", 50, 120, step=1,
        value=st.session_state.window_length_accel,
        key="window_length_accel",
        help="Fenstergröße Savitzky-Golay-Filter (größer = glatter, aber weniger Details).",
    )
accel_falling = not st.sidebar.toggle(
    "Falling / Rising", value=False,
    help="Falling = positive Beschleunigung (Weg/v steigt)\nRising = negative Beschleunigung (Weg/v nimmt ab)",
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
t_amax,       y_amax       = None, None
has_vmax = False
has_amax = False
v_max    = float('nan')
a_max    = float('nan')

if idx_end > idx_start:
    df_slice  = df.iloc[idx_start:idx_end + 1]
    arr       = df_slice[active_sensor].values
    dt_step_s = 1.0 / sample_rate

    # v-max: Spitzenwert der gefilterten Geschwindigkeit, gemittelt über Zeitbasis-Fenster
    gefilt_geschw_roh = _berechne_sg_ableitung(arr, dt_step_s, st.session_state.window_length, 1)
    if gefilt_geschw_roh is not None:
        gefilt_geschw = gefilt_geschw_roh / 1000.0   # µm/s → mm/s
        abs_geschw    = np.abs(gefilt_geschw)
        idx_vmax_peak = int(np.argmax(abs_geschw))
        iv_start      = max(0, idx_vmax_peak - halbes_zeitfenster)
        iv_ende       = min(len(arr) - 1, idx_vmax_peak + halbes_zeitfenster)
        v_max         = float(np.mean(abs_geschw[iv_start:iv_ende + 1]))

        abs_iv_start = idx_start + iv_start
        abs_iv_ende  = idx_start + iv_ende
        if 0 <= abs_iv_start <= max_idx and 0 <= abs_iv_ende <= max_idx:
            t_vmax_start = df.loc[abs_iv_start, 'Zeit (ms)']
            y_vmax_start = df.loc[abs_iv_start, active_sensor]
            t_vmax_ende  = df.loc[abs_iv_ende,  'Zeit (ms)']
            y_vmax_ende  = df.loc[abs_iv_ende,  active_sensor]
            has_vmax     = True

    # a-max: Spitzenwert der gefilterten Beschleunigung, gemittelt über Zeitbasis-Fenster
    gefilt_beschl_roh = _berechne_sg_ableitung(arr, dt_step_s, st.session_state.window_length_accel, 2)
    if gefilt_beschl_roh is not None:
        gefilt_beschl = gefilt_beschl_roh / 1_000_000.0   # µm/s² → m/s²
        idx_amax_peak = int(np.argmax(gefilt_beschl) if accel_falling else np.argmin(gefilt_beschl))
        ia_start      = max(0, idx_amax_peak - halbes_zeitfenster)
        ia_ende       = min(len(arr) - 1, idx_amax_peak + halbes_zeitfenster)
        a_max         = float(np.mean(gefilt_beschl[ia_start:ia_ende + 1]))

        abs_ia_mid = int(np.clip(idx_start + idx_amax_peak, 0, max_idx))
        t_amax     = float(df.loc[abs_ia_mid, 'Zeit (ms)'])
        y_amax     = float(df.loc[abs_ia_mid, active_sensor])
        has_amax   = True

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
fig.add_trace(go.Scatter(
    x=df_plot['Zeit (ms)'], y=df_plot[s1_name],
    name=s1_name, line=dict(color=FARBE_KANAL1),
))
fig.add_trace(go.Scatter(
    x=df_plot['Zeit (ms)'], y=df_plot[s2_name],
    name=s2_name, line=dict(color=FARBE_KANAL2),
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
if has_amax:
    fig.add_trace(go.Scatter(
        x=[t_amax], y=[y_amax],
        mode='markers', name='a-max',
        marker=dict(color=FARBE_AMAX, size=14, symbol='cross',
                    line=dict(color=FARBE_AMAX, width=2)),
    ))
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
y_max_plot   = float(df_plot[[s1_name, s2_name]].max().max())
y_min_plot   = float(df_plot[[s1_name, s2_name]].min().min())
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
            range=[-V_ACHSE_LIMIT_MM_S, V_ACHSE_LIMIT_MM_S],
        )
    )
if show_acceleration and acceleration is not None:
    fig.update_layout(
        yaxis3=dict(
            title='Beschleunigung (m/s²)',
            overlaying='y', side='right', showgrid=False,
            position=0.85 if show_velocity else 1.0,
            range=[-A_ACHSE_LIMIT_M_S2, A_ACHSE_LIMIT_M_S2],
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
    )
    st.slider(
        "XB", min_zeit, max_zeit, value=xb,
        key="xb_sw", step=0.001, format="%.3f ms",
        on_change=update_xb_from_slider, label_visibility="collapsed",
    )

# ---------------------------------------------------------------------------
# CROP / SHOW ALL
# ---------------------------------------------------------------------------

margin  = abs(xb - xa) * 0.15
crop_t0 = max(min_zeit, min(xa, xb) - margin)
crop_t1 = min(max_zeit, max(xa, xb) + margin)

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    if st.button("✂️ Crop A–B  (+15%)", disabled=(dt_val_ms == 0), width="stretch"):
        st.session_state.crop_start  = crop_t0
        st.session_state.crop_end    = crop_t1
        st.session_state.xa          = float(min(xa, xb))
        st.session_state.xb          = float(max(xa, xb))
        st.session_state.zoom_token += 1
        st.rerun()
with btn_col2:
    if st.button("🔍 Show All", disabled=not crop_active, width="stretch"):
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
a_label = "a-max Falling" if accel_falling else "a-max Rising"
hub_um  = abs(rect_fit['y_high'] - rect_fit['y_low']) if rect_fit is not None else float('nan')

r1, r2, r3, r4 = st.columns(4)
r1.metric("Δt (A-B)",         f"{dt_val_ms:.3f} ms")
r2.metric("v-mid (A-B)",      f"{v_avg:.1f} mm/s")
r3.metric("Δs (A-B)",         f"{dy_um:.1f} µm")
r4.metric("v-max (Peak)",     f"{v_max:.1f} mm/s"          if not np.isnan(v_max) else "N/A")

r5, r6, r7, r8 = st.columns(4)
r5.metric("Frequenz Δt (A-B)", f"{freq_hz:.1f} Hz"          if not np.isnan(freq_hz) else "N/A")
r6.metric("Δv Cursor (A B)",   f"{v_cursor_delta:.1f} mm/s" if not np.isnan(v_cursor_delta) else "N/A")
r7.metric("Hub Best-fit",      f"{hub_um:.1f} µm"           if not np.isnan(hub_um) else "N/A")
r8.metric(a_label,             f"{a_max:.0f} m/s²"          if not np.isnan(a_max) else "N/A")

# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------

st.sidebar.header("3. Export")
metrics = {
    "XA (ms)":              f"{xa:.3f}",
    "XB (ms)":              f"{xb:.3f}",
    "Δt (A-B)":             f"{dt_val_ms:.3f} ms",
    "Δs (A-B)":             f"{dy_um:.1f} µm",
    "v-mid (A-B)":          f"{v_avg:.1f} mm/s",
    "Hub Best-fit":         f"{hub_um:.1f} µm"           if not np.isnan(hub_um) else "N/A",
    "v-max (Peak)":         f"{v_max:.1f} mm/s"          if not np.isnan(v_max) else "N/A",
    "Frequenz Δt (A-B)":    f"{freq_hz:.1f} Hz"          if not np.isnan(freq_hz) else "N/A",
    "Δv Cursor (A B)":      f"{v_cursor_delta:.1f} mm/s" if not np.isnan(v_cursor_delta) else "N/A",
    a_label:                f"{a_max:.0f} m/s²"          if not np.isnan(a_max) else "N/A",
}
export_format = st.sidebar.radio(
    "Format:", ["PDF", "PNG"], horizontal=True, label_visibility="collapsed"
)
if st.sidebar.button("📥 Export erstellen", width="stretch"):
    with st.spinner("Wird erstellt..."):
        # Export nutzt den aktuell sichtbaren Bereich (Crop oder voll)
        chart_png = build_chart_png(
            df, s1_name, s2_name, active_sensor,
            xa, xb, ya, yb, show_v_avg,
            t_vmax_start, y_vmax_start, t_vmax_ende, y_vmax_ende, has_vmax,
            t_amax, y_amax, has_amax,
            show_rect_fit=show_rect_fit,
            rect_fit=rect_fit,
            show_velocity=show_velocity,
            window_length=st.session_state.window_length,
            show_acceleration=show_acceleration,
            window_length_accel=st.session_state.window_length_accel,
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
