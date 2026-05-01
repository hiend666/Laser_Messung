# pip install streamlit pandas plotly kaleido reportlab
# streamlit run app.py
import io
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

import datetime

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

# --- INITIALISIERUNG SESSION STATE ---
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
    # Crop-State: None = "Show All", sonst t_start / t_end als float
    'crop_start': None,
    'crop_end': None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# --- CSV LADEN MIT CACHE ---
@st.cache_data
def load_csv(file_bytes: bytes, skip_rows: int, ch1: str, ch2: str) -> pd.DataFrame:
    """Liest CSV im Roh- oder Sauber-Format und gibt DataFrame mit zwei Messspalten zurück."""
    probe = pd.read_csv(io.BytesIO(file_bytes), sep=',', decimal='.', header=None,
                        skiprows=skip_rows, nrows=1)
    first_cell = str(probe.iloc[0, 0]).strip()
    try:
        float(first_cell)
        df = pd.read_csv(io.BytesIO(file_bytes), sep=',', decimal='.',
                         header=None, skiprows=skip_rows)
        df = df.dropna(axis=1, how='all')
        data_cols = [c for c in df.columns if df[c].dtype in ['float64', 'float32']]
        if len(data_cols) < 2:
            raise ValueError("CSV enthält weniger als 2 numerische Spalten.")
        df = df[data_cols[:2]].copy()
        df.columns = [ch1, ch2]
    except ValueError as exc:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=',', decimal='.')
        df = df.dropna(axis=1, how='all')
        sensor_cols = [c for c in df.columns if c != 'Zeit (s)']
        if len(sensor_cols) < 2:
            raise ValueError(
                "CSV enthält weniger als 2 Messwert-Spalten nach dem Header."
            ) from exc
        df = df.rename(columns={sensor_cols[0]: ch1, sensor_cols[1]: ch2})
    return df


# --- ZEITACHSE (Modul-Ebene, Cache korrekt) ---
@st.cache_data
def build_time_axis(n_samples: int, sr: float) -> np.ndarray:
    return np.arange(n_samples) * (1000.0 / sr)   # ms


# --- OFFSET-ANWENDUNG MIT CACHE ---
@st.cache_data
def apply_offsets(
    raw_s1: np.ndarray, raw_s2: np.ndarray,
    off1: float, off2: float,
    zeit: np.ndarray,
    s1: str, s2: str,
) -> pd.DataFrame:
    """Erzeugt den verarbeiteten DataFrame – nur bei echten Änderungen neu berechnet."""
    return pd.DataFrame({
        'Zeit (ms)': zeit,
        s1: raw_s1 + off1,
        s2: raw_s2 + off2,
    })


# --- CALLBACKS ---
# Widgets schreiben immer in den freien Key (xa/xb), nie umgekehrt.
# Buttons/externe Setter schreiben nur in xa/xb – nie in Widget-Keys.
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


# --- SCHNELLE INDEX-BERECHNUNG (O(1)) ---
# x_val in ms, sample_rate in Hz → Index = x_ms / 1000 * sr
def get_idx_at_x(x_ms: float, sample_rate: float, max_idx: int) -> int:
    return int(np.clip(round(x_ms / 1000.0 * sample_rate), 0, max_idx))


# --- EXPORT: PLOTLY-DIAGRAMM ALS PNG ---
def build_chart_png(df, s1_name, s2_name, active_sensor,
                    xa, xb, ya, yb, show_v_avg,
                    t_vs, y_vs, t_ve, y_ve, has_vmax,
                    t_a_mid, y_a_mid, has_amax) -> bytes:
    # Y-Achse: 15 % über Maximalwert
    y_max_e = float(df[[s1_name, s2_name]].max().max())
    y_min_e = float(df[[s1_name, s2_name]].min().min())
    y_range_e = [y_min_e, y_max_e + (y_max_e - y_min_e) * 0.15]
    export_fig = go.Figure()
    export_fig.add_trace(go.Scatter(
        x=df['Zeit (ms)'], y=df[s1_name],
        name=s1_name, line=dict(color='#003366'),
    ))
    export_fig.add_trace(go.Scatter(
        x=df['Zeit (ms)'], y=df[s2_name],
        name=s2_name, line=dict(color='#4c78a8'),
    ))
    export_fig.add_vline(x=xa, line_dash="dash", line_color="red")
    export_fig.add_vline(x=xb, line_dash="dash", line_color="red")
    if show_v_avg:
        export_fig.add_trace(go.Scatter(
            x=[xa, xb], y=[ya, yb],
            mode='lines+markers', name='v-Schnitt',
            line=dict(color='green', width=2, dash='dot'),
        ))
    if has_vmax and t_vs is not None:
        export_fig.add_trace(go.Scatter(
            x=[t_vs, t_ve], y=[y_vs, y_ve],
            mode='lines+markers', name='v-max',
            line=dict(color='red', width=2),
        ))
    if has_amax and t_a_mid is not None:
        export_fig.add_trace(go.Scatter(
            x=[t_a_mid], y=[y_a_mid],
            mode='markers', name='a-max',
            marker=dict(color='orange', size=14, symbol='cross',
                        line=dict(color='orange', width=2)),
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
    return export_fig.to_image(format="png", width=1600, height=500, scale=2)


# --- EXPORT: PDF ---
def build_pdf(filename: str, chart_png: bytes, metrics: dict) -> bytes:
    buf = io.BytesIO()
    from reportlab.lib.pagesizes import landscape
    page = landscape(A4)          # 297 × 210 mm
    usable_w = page[0] - 30*mm   # 267 mm (je 15 mm Rand)
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
    dt_now   = datetime.datetime.now().strftime("%d.%m.%Y  %H:%M:%S")
    ts_style = ParagraphStyle(
        'ExportTS', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#666666'),
        fontName='Helvetica', alignment=2,   # 2 = RIGHT
    )
    header_tbl = Table(
        [[Paragraph("Messdaten-Auswertung", title_style),
          Paragraph(dt_now, ts_style)]],
        colWidths=[usable_w * 0.7, usable_w * 0.3],
        rowHeights=[8*mm],
    )
    header_tbl.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',  (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story = []
    story.append(header_tbl)
    story.append(Paragraph(f"Datei: {filename}", sub_style))
    img_buf = io.BytesIO(chart_png)
    img = Image(img_buf, width=usable_w, height=usable_w * 0.38)  # ~16:6 Seitenverhältnis
    story.append(img)
    story.append(Spacer(1, 4*mm))
    items      = list(metrics.items())
    half       = (len(items) + 1) // 2
    row1_items = items[:half]
    row2_items = items[half:]
    while len(row2_items) < half:
        row2_items.append(("", ""))

    col_width = usable_w / half
    col_widths = [col_width] * half

    def make_block(row_items):
        """Erstellt Label-Zeile + Werte-Zeile für eine Hälfte."""
        labels = [k for k, _ in row_items]
        values = [v for _, v in row_items]
        tbl = Table([labels, values], colWidths=col_widths, rowHeights=[7*mm, 8*mm])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',   (0, 0), (-1, 0), colors.HexColor('#003366')),
            ('TEXTCOLOR',    (0, 0), (-1, 0), colors.white),
            ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',     (0, 0), (-1, 0), 8),
            ('ALIGN',        (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND',   (0, 1), (-1, 1), colors.HexColor('#f0f4f8')),
            ('FONTNAME',     (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE',     (0, 1), (-1, 1), 9),
            ('TEXTCOLOR',    (0, 1), (-1, 1), colors.HexColor('#003366')),
            ('GRID',         (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ]))
        return tbl

    story.append(make_block(row1_items))
    story.append(Spacer(1, 2*mm))
    story.append(make_block(row2_items))
    doc.build(story)
    buf.seek(0)
    return buf.read()


# ===== HAUPTBEREICH =====

# --- SIDEBAR: IMPORT ---
st.sidebar.header("1. CSV-Import")
uploaded_file = st.sidebar.file_uploader("upload", type="csv", label_visibility="collapsed")

with st.sidebar.expander("⚙️ Einstellungen", expanded=not bool(uploaded_file)):
    sample_rate_unit = st.session_state.sample_rate_unit
    sample_rate_input = st.number_input(
        "Abtastung",
        min_value=0.0001,
        format="%.3f" if sample_rate_unit == "µs" else "%.1f",
        key="sample_rate",
        help="Hz = Abtastfrequenz, µs = Zeit pro Sample",
    )
    use_us = st.toggle(
        "Hz / µs",
        value=(sample_rate_unit == "µs"),
        key="sample_rate_unit_toggle",
        on_change=update_sample_rate_unit,
        label_visibility="visible",
    )
    sample_rate_unit = "µs" if use_us else "Hz"
    if st.session_state.sample_rate_unit != sample_rate_unit:
        st.session_state.sample_rate_unit = sample_rate_unit
    if sample_rate_unit == "Hz":
        sample_rate = sample_rate_input
    else:
        sample_rate = 1_000_000.0 / sample_rate_input

    st.number_input("Kopfzeilen überspringen", min_value=0, step=1, key="skip_rows")
    st.text_input("Kanal 1 Name", key="ch1_name")
    st.text_input("Kanal 2 Name", key="ch2_name")

if sample_rate <= 0:
    st.sidebar.error("Samplerate muss größer als 0 sein.")
    st.stop()

if uploaded_file:
    file_bytes = uploaded_file.read()
    ch1 = st.session_state.ch1_name or 'Kanal 1'
    ch2 = st.session_state.ch2_name or 'Kanal 2'

    try:
        df_raw = load_csv(file_bytes, st.session_state.skip_rows, ch1, ch2)
    except ValueError as e:
        st.error(f"Fehler beim Laden: {e}")
        st.stop()

    sensor_cols = [c for c in df_raw.columns if c != 'Zeit (s)']

    if len(sensor_cols) >= 2:
        s1_name, s2_name = sensor_cols[0], sensor_cols[1]

        # --- AUTO-RESET BEI NEUER DATEI ---
        if st.session_state.last_file_name != uploaded_file.name:
            total_time_ms = len(df_raw) / sample_rate * 1000.0  # ms
            off1_init  = float(df_raw[s1_name].min()) * -1.0
            off2_init  = float(df_raw[s2_name].min()) * -1.0
            st.session_state.off1        = off1_init
            st.session_state.off2        = off2_init
            st.session_state.off1_slider = off1_init
            st.session_state.off2_slider = off2_init
            st.session_state.xa  = total_time_ms * 0.30
            st.session_state.xb  = total_time_ms * 0.50
            st.session_state.crop_start  = None
            st.session_state.crop_end    = None
            st.session_state.zoom_token += 1
            st.session_state.last_file_name = uploaded_file.name
            st.rerun()

        # --- MANUELLE OFFSETS ---
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

        # --- DATENVERARBEITUNG ---
        zeit_full     = build_time_axis(len(df_raw), sample_rate)  # ms
        max_zeit_full = float(zeit_full[-1])
        max_idx_full  = len(df_raw) - 1

        df_full = apply_offsets(
            df_raw[s1_name].values, df_raw[s2_name].values,
            off1, off2, zeit_full, s1_name, s2_name,
        )

        # --- CROP-LOGIK ---
        crop_active = (
            st.session_state.crop_start is not None
            and st.session_state.crop_end is not None
        )
        if crop_active:
            c_start  = st.session_state.crop_start
            c_end    = st.session_state.crop_end
            ci_start = get_idx_at_x(c_start, sample_rate, max_idx_full)
            ci_end   = get_idx_at_x(c_end,   sample_rate, max_idx_full)
            df       = df_full.iloc[ci_start:ci_end + 1].reset_index(drop=True)
            min_zeit = float(df['Zeit (ms)'].iloc[0])
            max_zeit = float(df['Zeit (ms)'].iloc[-1])
            max_idx  = len(df) - 1
        else:
            df       = df_full
            min_zeit = 0.0
            max_zeit = max_zeit_full
            max_idx  = max_idx_full

        # --- SIDEBAR: AUSWERTUNG ---
        st.sidebar.header("2. Auswertung")
        active_sensor = st.sidebar.radio(
            "Kanal für Messung:", [s1_name, s2_name],
            horizontal=True, label_visibility="collapsed",
        )

        # Freie Cursor-Werte (ms) auf aktiven Zeitbereich clippen
        xa = float(np.clip(st.session_state.xa, min_zeit, max_zeit))
        xb = float(np.clip(st.session_state.xb, min_zeit, max_zeit))
        st.session_state.xa = xa
        st.session_state.xb = xb

        st.sidebar.number_input(
            "Zeit XA (ms)", min_zeit, max_zeit,
            value=xa, step=0.001, format="%.3f",
            key="xa_nw", on_change=update_xa_from_num,
        )
        st.sidebar.number_input(
            "Zeit XB (ms)", min_zeit, max_zeit,
            value=xb, step=0.001, format="%.3f",
            key="xb_nw", on_change=update_xb_from_num,
        )

        if xa > xb:
            st.sidebar.warning("⚠️ XA liegt nach XB – Marker vertauscht.")

        # Zeitbasis in ms (0.01 … 0.10 ms entspricht 10 … 100 µs bei 392 kHz)
        v_time_base_ms = st.sidebar.slider(
            "Zeitbasis v-max (ms)", 0.01, 0.10, 0.05,
            step=0.01, format="%.2f ms",
        )
        show_v_avg  = st.sidebar.toggle("v-Schnitt Linie (A-B) anzeigen", value=False)
        accel_falling = not st.sidebar.toggle("Falling / Rising", value=False,
                                          help="Falling = positive Beschleunigung (Weg/v steigt)\nRising = negative Beschleunigung (Weg/v nimmt ab)")

        # --- BERECHNUNGEN ---
        # xa/xb in ms; Crop-df hat ebenfalls ms-Zeitachse ab min_zeit
        if crop_active:
            idx_a = get_idx_at_x(xa - min_zeit, sample_rate, max_idx)
            idx_b = get_idx_at_x(xb - min_zeit, sample_rate, max_idx)
        else:
            idx_a = get_idx_at_x(xa, sample_rate, max_idx)
            idx_b = get_idx_at_x(xb, sample_rate, max_idx)

        ya = df.loc[idx_a, active_sensor]
        yb = df.loc[idx_b, active_sensor]

        dt_val_ms = abs(xb - xa)                          # ms
        dy_um     = abs(yb - ya)
        # v = Δy[µm→mm] / Δt[ms→s] = (dy/1000) / (dt_ms/1000) = dy/dt_ms
        v_avg = dy_um / dt_val_ms if dt_val_ms > 0 else 0.0  # mm/s

        # --- v-CURSOR: Momentangeschwindigkeit an XA und XB ---
        half_win = max(1, int(v_time_base_ms / 1000.0 * sample_rate / 2))

        def v_at_cursor(idx: int) -> float:
            i0 = max(0, idx - half_win)
            i1 = min(max_idx, idx + half_win)
            if i1 <= i0:
                return float('nan')
            dy = float(df.loc[i1, active_sensor] - df.loc[i0, active_sensor])
            dt_s = (i1 - i0) / sample_rate
            return (dy / 1000.0) / dt_s  # mm/s

        v_at_xa        = v_at_cursor(idx_a)
        v_at_xb        = v_at_cursor(idx_b)
        v_cursor_delta = abs(v_at_xb - v_at_xa) if not (np.isnan(v_at_xa) or np.isnan(v_at_xb)) else float('nan')

        idx_start, idx_end = sorted([idx_a, idx_b])

        t_vs, y_vs, t_ve, y_ve = None, None, None, None
        t_a_mid, y_a_mid       = None, None   # a-max Kreuz-Mittelpunkt
        has_vmax = False
        has_amax = False
        v_max    = float('nan')
        a_max    = float('nan')

        if idx_end > idx_start:
            df_slice    = df.iloc[idx_start:idx_end + 1]
            window_size = max(2, int(v_time_base_ms / 1000.0 * sample_rate))
            dt_step_s   = 1.0 / sample_rate          # s pro Sample
            dt_step_ms  = dt_step_s * 1000.0         # ms pro Sample
            arr         = df_slice[active_sensor].values

            if len(arr) > window_size:
                # diffs[i] = Durchschnittsgeschwindigkeit über Fenster i…i+window_size [mm/s]
                # Δy [µm→mm] / (window_size * dt [s])
                diffs     = (arr[window_size:] - arr[:-window_size]) / 1000.0 / (window_size * dt_step_s)
                abs_diffs = np.abs(diffs)
                n_diffs   = len(diffs)

                # --- v-max ---
                v_max      = float(np.max(abs_diffs))
                peak_rel   = int(np.argmax(abs_diffs))
                peak_end   = idx_start + window_size + peak_rel
                peak_start = peak_end - window_size
                if 0 <= peak_start <= max_idx and 0 <= peak_end <= max_idx:
                    t_vs = df.loc[peak_start, 'Zeit (ms)']
                    y_vs = df.loc[peak_start, active_sensor]
                    t_ve = df.loc[peak_end,   'Zeit (ms)']
                    y_ve = df.loc[peak_end,   active_sensor]
                    has_vmax = True

                # --- a-max: Δv zwischen zwei Fenstern im Abstand window_size ---
                # Vorzeichenbehaftete Beschleunigung [m/s²]:
                # Δv [mm/s] / (window_size * dt [s]) / 1000 → m/s²
                # Falling (positiv): Weg nimmt ab → dv < 0 → accel_signed < 0 → wir suchen Minimum
                # Rising  (negativ): Weg nimmt zu → dv > 0 → accel_signed > 0 → wir suchen Maximum
                if n_diffs > window_size:
                    accel_signed = (diffs[window_size:] - diffs[:-window_size]) / (window_size * dt_step_s) / 1000.0
                    if accel_falling:
                        # Falling: stärkste positive Beschleunigung (Weg/v steigt)
                        a_peak_i = int(np.argmax(accel_signed))
                        a_max    = float(accel_signed[a_peak_i])   # positiv
                    else:
                        # Rising: stärkste negative Beschleunigung (Weg/v nimmt ab)
                        a_peak_i = int(np.argmin(accel_signed))
                        a_max    = float(accel_signed[a_peak_i])   # negativ

                    # Kreuz-Mittelpunkt: Mitte zwischen den beiden Fenstern
                    mid_i   = int(np.clip(idx_start + a_peak_i + window_size, 0, max_idx))
                    t_a_mid = float(df.loc[mid_i, 'Zeit (ms)'])
                    y_a_mid = float(df.loc[mid_i, active_sensor])
                    has_amax = True

        # --- DOWNSAMPLING FÜR GROSSE DATEIEN ---
        MAX_PLOT_POINTS = 5000
        if len(df) > MAX_PLOT_POINTS:
            step    = len(df) // MAX_PLOT_POINTS
            df_plot = df.iloc[::step]
        else:
            df_plot = df

        # --- DIAGRAMM ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot['Zeit (ms)'], y=df_plot[s1_name],
            name=s1_name, line=dict(color='#003366'),
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['Zeit (ms)'], y=df_plot[s2_name],
            name=s2_name, line=dict(color='#4c78a8'),
        ))
        fig.add_vline(x=xa, line_dash="dash", line_color="red")
        fig.add_vline(x=xb, line_dash="dash", line_color="red")
        if show_v_avg:
            fig.add_trace(go.Scatter(
                x=[xa, xb], y=[ya, yb],
                mode='lines+markers', name='v-Schnitt',
                line=dict(color='green', width=2, dash='dot'),
            ))
        if has_vmax:
            fig.add_trace(go.Scatter(
                x=[t_vs, t_ve], y=[y_vs, y_ve],
                mode='lines+markers', name='v-max',
                line=dict(color='red', width=2),
            ))
        if has_amax:
            fig.add_trace(go.Scatter(
                x=[t_a_mid], y=[y_a_mid],
                mode='markers', name='a-max',
                marker=dict(color='orange', size=14, symbol='cross',
                            line=dict(color='orange', width=2)),
            ))

        # Y-Achse: 15 % über Maximalwert damit Legende den Graph nicht verdeckt
        y_max_plot = float(df_plot[[s1_name, s2_name]].max().max())
        y_min_plot = float(df_plot[[s1_name, s2_name]].min().min())
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
        st.plotly_chart(fig, width="stretch", key="main_chart")

        # --- CURSOR-SLIDER (ms) ---
        # 0.04-Spalte gleicht Y-Achsen-Breite aus damit Slider mit Diagramm fluchten
        c_pad, c_slider = st.columns([0.04, 0.96])
        with c_slider:
            st.slider(
                "XA", min_zeit, max_zeit,
                value=xa,
                key="xa_sw", step=0.001, format="%.3f ms",
                on_change=update_xa_from_slider,
                label_visibility="collapsed",
            )
            st.slider(
                "XB", min_zeit, max_zeit,
                value=xb,
                key="xb_sw", step=0.001, format="%.3f ms",
                on_change=update_xb_from_slider,
                label_visibility="collapsed",
            )

        # --- CROP / SHOW ALL ---
        margin  = abs(xb - xa) * 0.15
        crop_t0 = max(min_zeit, min(xa, xb) - margin)
        crop_t1 = min(max_zeit, max(xa, xb) + margin)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("✂️ Crop A–B  (+15%)", disabled=(dt_val_ms == 0), width="stretch"):
                st.session_state.crop_start  = crop_t0
                st.session_state.crop_end    = crop_t1
                new_xa = float(min(xa, xb))
                new_xb = float(max(xa, xb))
                st.session_state.xa = new_xa
                st.session_state.xb = new_xb
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

        # --- ERGEBNISSE ---
        freq_hz = (1000.0 / dt_val_ms) if dt_val_ms > 0 else float('nan')
        a_label = "a-max Falling" if accel_falling else "a-max Rising"

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Δt (A-B)",           f"{dt_val_ms:.3f} ms")
        r2.metric("v-mid (A-B)",         f"{v_avg:.1f} mm/s")
        r3.metric("Δs (A-B)",           f"{dy_um:.1f} µm")
        r4.metric("v-max (Peak)",        f"{v_max:.1f} mm/s"          if not np.isnan(v_max) else "N/A")

        r5, r6, r7, r8 = st.columns(4)
        r5.metric("Frequenz Δt (A-B)",   f"{freq_hz:.1f} Hz"          if not np.isnan(freq_hz) else "N/A")
        r6.metric("Δv Cursor (A B)",     f"{v_cursor_delta:.1f} mm/s" if not np.isnan(v_cursor_delta) else "N/A")
        r8.metric(a_label,               f"{a_max:.0f} m/s²"          if not np.isnan(a_max) else "N/A")

        # --- EXPORT ---
        st.sidebar.header("3. Export")
        metrics = {
            "XA (ms)":              f"{xa:.3f}",
            "XB (ms)":              f"{xb:.3f}",
            "Δt (A-B)":             f"{dt_val_ms:.3f} ms",
            "Δs (A-B)":             f"{dy_um:.1f} µm",
            "v-mid (A-B)":          f"{v_avg:.1f} mm/s",
            "v-max (Peak)":         f"{v_max:.1f} mm/s"        if not np.isnan(v_max) else "N/A",
            "Frequenz Δt (A-B)":    f"{freq_hz:.1f} Hz"        if not np.isnan(freq_hz) else "N/A",
            "Δv Cursor (A B)":      f"{v_cursor_delta:.1f} mm/s" if not np.isnan(v_cursor_delta) else "N/A",
            a_label:                f"{a_max:.0f} m/s²"        if not np.isnan(a_max) else "N/A",
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
                    t_vs, y_vs, t_ve, y_ve, has_vmax,
                    t_a_mid, y_a_mid, has_amax,
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

    else:
        st.error("Die Datei benötigt mindestens zwei Messwert-Spalten.")

else:
    st.info("Bitte laden Sie eine CSV-Datei hoch, um die Analyse zu starten.")
