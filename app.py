# pip install streamlit pandas plotly scipy "kaleido==0.2.1" reportlab
# streamlit run app.py
"""
Messdaten-Auswertung – Laservibrometer CSV.

Zeigt Signal-, D- und D2-Kurven (1. und 2. Ableitung) und
exportiert Ergebnisse als PDF oder PNG.
"""
import json

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

import reader
from chart import (
    KANAL_FARBEN, FARBE_D, FARBE_D2, FARBE_V_SCHNITT,
    FARBE_VMAX, FARBE_AMAX, FARBE_CURSOR,
    _ZEIT_TO_S, _ableit_info, _yachsen_layout,
    _zeichne_rechteck_fit, _zeichne_integral_flaeche, _finde_sop_kreuzungen,
    _baue_traces, build_chart_png, build_pdf,
)

VERSION = "v1.01.01"

# ---------------------------------------------------------------------------
# KONSTANTEN
# ---------------------------------------------------------------------------

MAX_PLOT_PUNKTE = 5_000     # Downsampling-Schwelle für interaktives Diagramm
N_KANÄLE        = 4         # Maximale Kanalanzahl – einzige Stelle um diese zu ändern
Y_PUFFER        = 0.15      # Y-Bereich-Puffer oben (15 %) – muss mit chart.Y_PUFFER übereinstimmen

# Einheiten-Auswahl für Kanäle
EINHEIT_OPTIONEN = ['µm', 'mm', 'm', 'V', 'mV', 'A', 'mA', 'N', 'kN', 'bar', 'Pa', '°C', '%']
EINHEIT_ALLE     = ['µm', 'mm', 'm', 'V', 'mV', 'A', 'mA', 'N', 'kN', 'bar', 'Pa', '°C', '%']

# Cursor-Startposition beim Laden einer neuen Datei (Anteil der Gesamtzeit)
CURSOR_XA_INIT_FRAC = 0.30
CURSOR_XB_INIT_FRAC = 0.60

# Y-Offset-Slider: Mindestbereich und Skalierungsfaktor für automatische Grenzen
Y_OFFSET_LIMIT_MIN    = 600.0
Y_OFFSET_LIMIT_FAKTOR = 1.5

# Schrittweiten-Schwellen für Y-Offset-Slider (Absolutwert → Schrittweite)
SLIDER_STEP_SCHWELLEN = (600.0, 6000.0, 60000.0)

# Rand beim Crop (Anteil des Cursor-Abstands beiderseits)
CROP_RAND_FAKTOR = 0.15

# k-Means Iterationen für Rechteck-Fit-Schwellwert-Berechnung
K_MEANS_ITER = 5

# Zeiteinheit-Lookup: hz_faktor (int) → Einheitenstring, abgeleitet aus _ZEIT_TO_S
_ZEIT_HZ_ZU_EINHEIT: dict[int, str] = {round(1 / v): k for k, v in _ZEIT_TO_S.items()}


def _einh_sfx(einheit: str) -> str:
    return einheit.replace('µ', 'u').replace('°', 'deg').replace('%', 'pct').replace('/', 'p')

def einheit_ss_key_min(einheit: str) -> str:
    return 'ymin_' + _einh_sfx(einheit)

def einheit_ss_key_max(einheit: str) -> str:
    return 'ymax_' + _einh_sfx(einheit)

# ---------------------------------------------------------------------------
# STREAMLIT-SEITENKONFIGURATION UND CSS
# ---------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="VERMESsdaten Auswertung")
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
_CH_NAMEN_DEFAULT  = ['Festo', 'DST'] + [''] * (N_KANÄLE - 2)
_OSC_SKALE_DEFAULT = [1.0, 1.0, 100.0] + [1.0] * (N_KANÄLE - 3)

defaults = {
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
    'max_samples': 8000,
    # Crop-State: None = "Show All", sonst t_start / t_end als float
    'crop_start': None,
    'crop_end': None,
    'show_v_avg': False,
    'show_rect_fit': False,
    'show_velocity': False,
    'window_length': 30,
    'show_acceleration': False,
    'window_length_accel': 40,
    'show_sop': False,
    'sop_percent': 80,
    'show_integral': False,
    'v_axis_min': -3_200,
    'v_axis_max':  3_200,
    'a_axis_min': -20_000,
    'a_axis_max':  20_000,
    'zeit_hz_faktor': 1000.0,  # Umrechnungsfaktor s → Anzeigeeinheit (1e3=ms, 1e6=µs, 1e9=ns)
    'n_kanäle_datei': N_KANÄLE,
    'sub_dateityp':   True,
    'sub_einlesen':   False,
    'sub_kanaele':    False,
    'sub_offsets':    False,
    'sub_xoffset':    False,
    'sub_grenzwerte': False,
    'sub_speichern':  False,
    'einstellungen':  True,
}
for _i in range(1, N_KANÄLE + 1):
    defaults[f'ch{_i}_name']    = _CH_NAMEN_DEFAULT[_i - 1]
    defaults[f'ch{_i}_einheit'] = 'µm'
    defaults[f'osc_skale_{_i}'] = _OSC_SKALE_DEFAULT[_i - 1]
    defaults[f'off{_i}']        = 0.0
    defaults[f'off{_i}_slider'] = 0.0
    defaults[f'x_off{_i}']     = 0.0
    defaults[f'show_ch{_i}']   = True
for _e in EINHEIT_ALLE:
    defaults[einheit_ss_key_min(_e)] = 0.0
    defaults[einheit_ss_key_max(_e)] = 0.0
for _i in range(1, N_KANÄLE + 1):
    defaults[f'ch{_i}_ymin'] = 0.0
    defaults[f'ch{_i}_ymax'] = 0.0

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

EINSTELLUNGEN_KEYS: list[str] = [
    'file_type_radio',
    'sample_rate', 'sample_rate_unit', 'sample_rate_unit_toggle',
    'skip_rows', 'max_samples',
    'xa', 'xb',
    'show_v_avg', 'show_rect_fit',
    'show_velocity', 'window_length',
    'show_acceleration', 'window_length_accel',
    'show_sop', 'sop_percent',
    'show_integral',
    'v_axis_min', 'v_axis_max', 'a_axis_min', 'a_axis_max',
]
for _i in range(1, N_KANÄLE + 1):
    EINSTELLUNGEN_KEYS += [
        f'ch{_i}_name', f'ch{_i}_einheit', f'osc_skale_{_i}',
        f'off{_i}', f'off{_i}_slider', f'x_off{_i}', f'show_ch{_i}',
        f'ch{_i}_ymin', f'ch{_i}_ymax',
    ]
for _e in EINHEIT_ALLE:
    EINSTELLUNGEN_KEYS.append(einheit_ss_key_min(_e))
    EINSTELLUNGEN_KEYS.append(einheit_ss_key_max(_e))


# ---------------------------------------------------------------------------
# GECACHTE DATENFUNKTIONEN  (Wrapper um reader.py – Streamlit-Caching hier)
# ---------------------------------------------------------------------------


@st.cache_data
def load_rohdaten(
    file_bytes: bytes,
    file_type: str,
    skip_rows: int,
    max_samples: int,
    kanal_namen: tuple[str, ...],
    kanal_skalierung: tuple[float, ...] = (),
) -> tuple[pd.DataFrame, float]:
    """Gecachter Wrapper um reader.load_raw – wird nur bei Dateiänderung neu ausgeführt.
    Gibt (DataFrame, hz_faktor) zurück."""
    return reader.load_raw(file_bytes, file_type, skip_rows, max_samples, kanal_namen, kanal_skalierung)


@st.cache_data
def _detect_kanal_count_cached(file_bytes: bytes, file_type: str, skip_rows: int = 0) -> int:
    """Gecachter Wrapper um reader.detect_kanal_count."""
    return reader.detect_kanal_count(file_bytes, file_type, skip_rows)


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

    for _ in range(K_MEANS_ITER):
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


@st.cache_data
def _build_display_df_cached(
    df_raw: pd.DataFrame,
    file_type: str,
    sample_rate_hz: float,
    kanal_namen: tuple[str, ...],
    offsets: tuple[float, ...],
    zeit_hz_faktor: float = 1000.0,
) -> tuple[pd.DataFrame, float]:
    """Gecachter Wrapper um reader.build_display_df – Y-Offsets und Zeitachse."""
    return reader.build_display_df(df_raw, file_type, sample_rate_hz, kanal_namen, offsets, zeit_hz_faktor)


# ---------------------------------------------------------------------------
# CALLBACKS – Zwei-Key-Muster
# Widgets schreiben immer in den freien Key (xa/xb/off1…4), nie umgekehrt.
# ---------------------------------------------------------------------------

def _make_cursor_cb(cursor: str, source: str):
    """Erzeugt Callback für Cursor-Widget (slider oder num).
    Schreibt den Wert des auslösenden Widgets in den freien Key und das Peer-Widget.
    """
    sw_key   = f'{cursor}_sw'
    nw_key   = f'{cursor}_nw'
    from_key = sw_key if source == 'slider' else nw_key
    to_key   = nw_key if source == 'slider' else sw_key
    def _cb():
        val = max(0.0, float(st.session_state[from_key]))
        st.session_state[cursor] = val
        st.session_state[to_key] = val
    return _cb

update_xa_from_slider = _make_cursor_cb('xa', 'slider')
update_xa_from_num    = _make_cursor_cb('xa', 'num')
update_xb_from_slider = _make_cursor_cb('xb', 'slider')
update_xb_from_num    = _make_cursor_cb('xb', 'num')

def _make_off_cb(i: int):
    def _cb(): st.session_state[f'off{i}'] = st.session_state[f'off{i}_slider']
    return _cb

OFF_CALLBACKS = [_make_off_cb(i) for i in range(1, N_KANÄLE + 1)]


def _setze_cursor_position(total_time_ms: float) -> None:
    """Setzt XA und XB auf Standardposition (30 % und 60 % der Gesamtzeit)."""
    xa_init = total_time_ms * CURSOR_XA_INIT_FRAC
    xb_init = total_time_ms * CURSOR_XB_INIT_FRAC
    st.session_state.xa    = xa_init
    st.session_state.xa_sw = xa_init
    st.session_state.xa_nw = xa_init
    st.session_state.xb    = xb_init
    st.session_state.xb_sw = xb_init
    st.session_state.xb_nw = xb_init


def _fmt_zeit(value: float, einheit: str) -> str:
    """Formatiert Zeitwert mit automatisch gewählter Einheit (s/ms/µs/ns).
    Wechselt die Einheit wenn mehr als 3 Dezimal- oder Vorkommastellen nötig.
    """
    val_s = value * _ZEIT_TO_S.get(einheit, 1e-3)
    if val_s == 0:
        return f"0.000 {einheit}"
    for unit in ('s', 'ms', 'µs', 'ns'):
        scaled = abs(val_s) / _ZEIT_TO_S[unit]
        if scaled >= 0.001:
            prec = ".0f" if scaled >= 1000 else ".1f" if scaled >= 100 else ".2f" if scaled >= 10 else ".3f"
            return f"{val_s / _ZEIT_TO_S[unit]:{prec}} {unit}"
    return f"{val_s / _ZEIT_TO_S['ns']:.3f} ns"


def _fmt_freq(value_hz: float) -> str:
    """Formatiert Frequenz mit automatisch gewählter Einheit (Hz/kHz/MHz/GHz)."""
    if np.isnan(value_hz) or value_hz <= 0:
        return "N/A"
    abs_hz = abs(value_hz)
    if abs_hz >= 1e9:
        return f"{value_hz/1e9:.3f} GHz"
    if abs_hz >= 1e6:
        return f"{value_hz/1e6:.3f} MHz"
    if abs_hz >= 1e3:
        return f"{value_hz/1e3:.3f} kHz"
    return f"{value_hz:.3f} Hz"


def _fmt_val(value: float, einheit: str, prec: int = 3) -> str:
    """Formatiert Messwert: bei sehr kleinen/großen Absolutwerten SI-Prefix wählen."""
    if np.isnan(value):
        return "N/A"
    abs_v = abs(value)
    if abs_v == 0 or (0.001 <= abs_v < 10_000):
        return f"{value:.{prec}f} {einheit}"
    return f"{value:.{prec}e} {einheit}"


def _fmt_integral(value_raw: float, aktiv_einheit: str, zeit_einheit: str) -> str:
    """Formatiert Integralwert (aktiv_einheit × zeit_einheit) mit auto-skalierter Zeiteinheit."""
    if np.isnan(value_raw):
        return "N/A"
    val_si = value_raw * _ZEIT_TO_S.get(zeit_einheit, 1e-3)   # → aktiv_einheit × s
    if val_si == 0:
        return f"0.000 {aktiv_einheit}·s"
    abs_si = abs(val_si)
    for unit in ('s', 'ms', 'µs', 'ns'):
        scaled = abs_si / _ZEIT_TO_S[unit]
        if scaled >= 0.001:
            prec = ".0f" if scaled >= 1000 else ".1f" if scaled >= 100 else ".2f" if scaled >= 10 else ".3f"
            return f"{val_si / _ZEIT_TO_S[unit]:{prec}} {aktiv_einheit}·{unit}"
    return f"{val_si / _ZEIT_TO_S['ns']:.3f} {aktiv_einheit}·ns"


def update_sample_rate_unit():
    new_unit = "µs" if st.session_state.sample_rate_unit_toggle else "Hz"
    old_unit = st.session_state.sample_rate_unit
    if new_unit != old_unit:
        if st.session_state.sample_rate > 0:
            st.session_state.sample_rate = 1_000_000.0 / st.session_state.sample_rate
        st.session_state.sample_rate_unit = new_unit


_SUB_EXPANDER_KEYS = (
    'sub_dateityp', 'sub_einlesen', 'sub_kanaele',
    'sub_offsets', 'sub_xoffset', 'sub_grenzwerte', 'sub_speichern',
)


def _make_sub_expander_cb(this_key: str):
    """Akkordeon-Callback: öffnet der Nutzer diesen Expander, werden alle anderen geschlossen.
    Beim Schließen von 'sub_kanaele' werden leere Kanalnamen automatisch belegt."""
    def _cb():
        if st.session_state.get(this_key, False):
            for _k in _SUB_EXPANDER_KEYS:
                if _k != this_key:
                    st.session_state[_k] = False
        if this_key == 'sub_kanaele' and not st.session_state.get('sub_kanaele', False):
            _n = st.session_state.get('n_kanäle_datei', N_KANÄLE)
            for _i in range(1, _n + 1):
                if not st.session_state.get(f'ch{_i}_name', '').strip():
                    st.session_state[f'ch{_i}_name'] = f'Kanal {_i}'
    return _cb

_SUB_EXPANDER_CBS = {k: _make_sub_expander_cb(k) for k in _SUB_EXPANDER_KEYS}


def on_file_upload():
    """Schließt Einstellungen-Expander beim Hochladen einer neuen Datei."""
    if st.session_state.get('_file_uploader') is not None:
        st.session_state.einstellungen = False
        for _k in _SUB_EXPANDER_KEYS:
            st.session_state[_k] = False


def on_settings_upload():
    """Liest hochgeladene JSON-Einstellungen und schreibt sie in session_state.

    Läuft als on_change-Callback vor dem Widget-Rendering – daher dürfen
    hier alle Keys gesetzt werden, auch solche die Widget-Keys sind.
    """
    f = st.session_state.get('_settings_uploader')
    if f is None:
        st.session_state['_settings_load_status'] = None
        return
    try:
        _loaded: dict = json.loads(f.read())
        for _k, _v in _loaded.items():
            if _k in EINSTELLUNGEN_KEYS:
                st.session_state[_k] = _v
        st.session_state['_settings_load_status'] = 'ok'
    except Exception as _exc:
        st.session_state['_settings_load_status'] = str(_exc)


_DATEITYP_KANAELE: dict[str, list[dict]] = {
    "Hubmessung": [
        {'name': 'Hub',      'einheit': 'µm', 'skale': 1.0},
    ],
    "CSV plain": [
        {'name': 'Festo',    'einheit': 'µm', 'skale': 1.0},
        {'name': 'DST',      'einheit': 'µm', 'skale': 1.0},
    ],
    "Oszilloskop CSV": [
        {'name': 'Strom',    'einheit': 'A',  'skale': 1.0},
        {'name': 'Spannung', 'einheit': 'V',  'skale': 1.0},
        {'name': 'Weg',      'einheit': 'µm', 'skale': 100.0},
    ],
}


def update_sample_rate_for_file_type():
    """Setzt Samplerate und Kanal-Defaults basierend auf dem Dateityp."""
    ft = st.session_state.get('file_type_radio', 'CSV plain')
    if ft == "Hubmessung":
        st.session_state.sample_rate = 2.55
        st.session_state.sample_rate_unit = "µs"
        st.session_state.sample_rate_unit_toggle = True
    preset = _DATEITYP_KANAELE.get(ft, [])
    for i in range(1, N_KANÄLE + 1):
        cfg = preset[i - 1] if i - 1 < len(preset) else {}
        st.session_state[f'ch{i}_name']    = cfg.get('name', '')
        st.session_state[f'ch{i}_einheit'] = cfg.get('einheit', 'µm')
        st.session_state[f'osc_skale_{i}'] = cfg.get('skale', 1.0)
        st.session_state[f'show_ch{i}']    = True


# ---------------------------------------------------------------------------
# HILFSFUNKTIONEN – INDEX UND ABLEITUNGEN
# ---------------------------------------------------------------------------

def get_idx_at_x(x: float, sample_rate: float, max_idx: int, hz_faktor: float = 1000.0) -> int:
    """Wandelt Zeitwert (in Anzeigeeinheit) in DataFrame-Index um. O(1)."""
    return int(np.clip(round(x / hz_faktor * sample_rate), 0, max_idx))


# ---------------------------------------------------------------------------
# HAUPTBEREICH
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SIDEBAR: CSV-IMPORT UND EINSTELLUNGEN
# ---------------------------------------------------------------------------

st.sidebar.header("1. Import")
st.sidebar.caption(f"Version: {VERSION}")

file_type = st.session_state.get('file_type_radio', 'CSV plain')
_hz_faktor    = st.session_state.get('zeit_hz_faktor', 1000.0)
_zeit_einheit = _ZEIT_HZ_ZU_EINHEIT.get(round(_hz_faktor), 'ms')

if file_type == "Hubmessung":
    file_extensions = ["txt"]
else:
    file_extensions = ["csv"]

uploaded_file = st.sidebar.file_uploader(
    "upload", type=file_extensions, label_visibility="collapsed",
    key="_file_uploader",
    on_change=on_file_upload,
    help="Datei hochladen. CSV plain / Oszilloskop CSV: .csv-Datei. Hubmessung: .txt-Datei.",
)


with st.sidebar.expander("Einstellungen", expanded=st.session_state.einstellungen, key="einstellungen"):

    with st.expander("Dateityp", expanded=st.session_state.sub_dateityp, key="sub_dateityp", on_change=_SUB_EXPANDER_CBS['sub_dateityp']):
        st.radio(
            "Dateityp",
            ["CSV plain", "Hubmessung", "Oszilloskop CSV"],
            key="file_type_radio",
            help="CSV plain: Komma-getrennt ohne Zeitachse. Hubmessung: TAB-getrennt mit Zeitachse. Oszilloskop CSV: Komma-getrennt mit Zeitachse in Sekunden.",
            on_change=update_sample_rate_for_file_type,
        )

    with st.expander("Einlesen", expanded=st.session_state.sub_einlesen, key="sub_einlesen", on_change=_SUB_EXPANDER_CBS['sub_einlesen']):
        if file_type == "Oszilloskop CSV":
            st.caption("Zeitachse wird aus der Datei gelesen.")
            sample_rate_input = st.session_state.sample_rate
            sample_rate_unit  = st.session_state.sample_rate_unit
            sample_rate       = 1_000_000.0 / sample_rate_input if sample_rate_unit == "µs" else sample_rate_input
        else:
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

        if file_type == "CSV plain":
            st.number_input("Kopfzeilen überspringen", min_value=0, step=1, key="skip_rows",
                            help="Anzahl der Zeilen am Dateianfang die ignoriert werden (z. B. Metadaten-Header).")
        st.number_input("Max. Samples importieren", min_value=0, step=1000, key="max_samples",
                        help="Maximale Anzahl der zu importierenden Datenpunkte (0 = alle importieren).")

    _n_show = N_KANÄLE
    if uploaded_file:
        try:
            _n_show = min(N_KANÄLE, max(1, _detect_kanal_count_cached(
                uploaded_file.getvalue(), file_type,
                st.session_state.get('skip_rows', 12),
            )))
        except Exception:
            pass
        st.session_state['n_kanäle_datei'] = _n_show

    if uploaded_file and _n_show > 0:
        for _i in range(1, _n_show + 1):
            if not st.session_state.get(f'ch{_i}_name', '').strip():
                st.session_state[f'ch{_i}_name'] = f'Kanal {_i}'

    with st.expander("Kanäle", expanded=st.session_state.sub_kanaele, key="sub_kanaele", on_change=_SUB_EXPANDER_CBS['sub_kanaele']):
        st.caption("Leer: wird beim Schließen automatisch als 'Kanal N' benannt.")
        _osc_einheiten: list[str] = []
        if file_type == "Oszilloskop CSV" and uploaded_file:
            _, _osc_einheiten = reader.peek_oszilloskop_header(uploaded_file.getvalue())
        for _ki in range(_n_show):
            _i         = _ki + 1
            _ch_key    = f'ch{_i}_name'
            _einh_key  = f'ch{_i}_einheit'
            _skale_key = f'osc_skale_{_i}'
            _osc_hint  = f" [{_osc_einheiten[_ki]}]" if _ki < len(_osc_einheiten) else ""
            _cur_einh  = st.session_state.get(_einh_key, 'µm')
            _einh_opts = EINHEIT_OPTIONEN if _cur_einh in EINHEIT_OPTIONEN else [_cur_einh] + EINHEIT_OPTIONEN
            _col_name, _col_skale, _col_einh = st.columns([2, 1, 1])
            _col_name.text_input(
                f"Kanal {_i}{_osc_hint}", key=_ch_key,
                max_chars=12,
                help="Leer lassen um diesen Kanal nicht einzulesen.",
            )
            if file_type == "Oszilloskop CSV":
                _col_skale.number_input(
                    "×",
                    step=0.01, format="%.2f", key=_skale_key,
                    help="Skalierungsfaktor (Rohwert × Faktor).",
                )
            _col_einh.selectbox(
                "Einheit", _einh_opts, key=_einh_key,
                help="Physikalische Einheit – bestimmt die Y-Achse.",
            )

    if uploaded_file:
        _kanal_cfg = [st.session_state.get(f'ch{i}_name', '').strip() for i in range(1, N_KANÄLE + 1)]
        # Nur die ersten _n_show Kanäle verwenden – verhindert, dass alte Namen aus
        # früheren Dateien mit mehr Kanälen mitgeschleppt werden.
        kanal_namen_tuple = tuple(n for n in _kanal_cfg[:_n_show] if n)
        _sensor_ch_num    = {name: i + 1 for i, name in enumerate(_kanal_cfg[:_n_show]) if name}

        if len(kanal_namen_tuple) < 1:
            st.sidebar.error("Mindestens ein Kanalname muss angegeben werden.")
            st.stop()

        sensor_namen = list(kanal_namen_tuple)
        file_bytes = uploaded_file.getvalue()
        _osc_skale_tuple = tuple(
            st.session_state[f'osc_skale_{i+1}'] for i in range(len(kanal_namen_tuple))
        )

        try:
            df_raw, _hz_f_datei = load_rohdaten(
                file_bytes, file_type, st.session_state.skip_rows,
                st.session_state.max_samples, kanal_namen_tuple, _osc_skale_tuple,
            )
            st.session_state['zeit_hz_faktor'] = _hz_f_datei
        except ValueError as e:
            st.error(f"Fehler beim Laden: {e}")
            st.stop()

        if st.session_state.last_file_name != uploaded_file.name:
            _hz_f_init = st.session_state.get('zeit_hz_faktor', 1000.0)
            if file_type in ("Hubmessung", "Oszilloskop CSV"):
                total_time_ms = float(df_raw['Zeit (ms)'].iloc[-1])
            else:
                total_time_ms = len(df_raw) / sample_rate * _hz_f_init
            for i, name in enumerate(sensor_namen, 1):
                off_init = float(df_raw[name].min()) * -1.0
                st.session_state[f'off{i}']        = off_init
                st.session_state[f'off{i}_slider'] = off_init
            for i in range(len(sensor_namen) + 1, N_KANÄLE + 1):
                st.session_state[f'off{i}']        = 0.0
                st.session_state[f'off{i}_slider'] = 0.0
            _n_d = st.session_state.get('n_kanäle_datei', N_KANÄLE)
            for _i in range(_n_d + 1, N_KANÄLE + 1):
                st.session_state[f'ch{_i}_name'] = ''
            for _i in range(1, _n_show + 1):
                st.session_state[f'show_ch{_i}'] = True
            _setze_cursor_position(total_time_ms)
            st.session_state.crop_start     = None
            st.session_state.crop_end       = None
            st.session_state.zoom_token    += 1
            st.session_state.last_file_name = uploaded_file.name
            st.session_state.einstellungen  = False
            st.rerun()

        offs = tuple(st.session_state[f'off{i+1}'] for i in range(len(sensor_namen)))
    else:
        df_raw = None
        sensor_namen = []
        offs = tuple()

    with st.expander("Y-Offset", expanded=st.session_state.sub_offsets, key="sub_offsets", on_change=_SUB_EXPANDER_CBS['sub_offsets']):
        if uploaded_file and df_raw is not None and sensor_namen:
            with st.container(border=True):
                st.subheader("Auf 0 setzen")
                n_ch     = len(sensor_namen)
                btn_cols = st.columns(n_ch)
                for i, name in enumerate(sensor_namen):
                    if btn_cols[i].button(f"{name}", use_container_width=True,
                                          help="Setzt den Y-Offset so, dass der Minimalwert des Kanals auf 0 liegt.", key=f"auto0_{i}"):
                        val = float(df_raw[name].min()) * -1.0
                        st.session_state[f'off{i+1}']        = val
                        st.session_state[f'off{i+1}_slider'] = val
                        st.rerun()

            if st.button("↺ Reset (alle auf 0)", key="reset_offsets",
                         help="Setzt alle Y-Offsets auf 0.", use_container_width=True):
                for _ri in range(1, N_KANÄLE + 1):
                    st.session_state[f'off{_ri}']        = 0.0
                    st.session_state[f'off{_ri}_slider'] = 0.0
                st.rerun()

            st.markdown("")
            for i, name in enumerate(sensor_namen):
                _raw_col  = df_raw[name]
                _off_lim  = max(Y_OFFSET_LIMIT_MIN,
                                abs(float(_raw_col.min())) * Y_OFFSET_LIMIT_FAKTOR,
                                abs(float(_raw_col.max())) * Y_OFFSET_LIMIT_FAKTOR)
                _off_lim  = float(np.ceil(_off_lim / 100) * 100)
                _off_step = (0.1   if _off_lim <= SLIDER_STEP_SCHWELLEN[0] else
                             1.0   if _off_lim <= SLIDER_STEP_SCHWELLEN[1] else
                             10.0  if _off_lim <= SLIDER_STEP_SCHWELLEN[2] else 100.0)
                st.slider(
                    name, -_off_lim, _off_lim, step=_off_step,
                    key=f'off{i+1}_slider', on_change=OFF_CALLBACKS[i],
                    help=f"Y-Versatz für diesen Kanal (Bereich ±{_off_lim:.0f}).",
                )

    with st.expander("X-Offset", expanded=st.session_state.sub_xoffset, key="sub_xoffset", on_change=_SUB_EXPANDER_CBS['sub_xoffset']):
        if uploaded_file and sensor_namen:
            for _xi, _xname in enumerate(sensor_namen):
                st.number_input(
                    f"X-Offset {_xname} ({_zeit_einheit})",
                    step=0.1, format="%.3f",
                    key=f'x_off{_xi+1}',
                    help=f"Zeitversatz in {_zeit_einheit} – verschiebt diesen Kanal nach links (−) oder rechts (+).",
                )

    with st.expander("Diagramm-Grenzwerte", expanded=st.session_state.sub_grenzwerte, key="sub_grenzwerte", on_change=_SUB_EXPANDER_CBS['sub_grenzwerte']):
        st.caption("min / max  (0 = automatisch)")
        for _i in range(1, N_KANÄLE + 1):
            _ch_name = st.session_state.get(f'ch{_i}_name', '').strip()
            if not _ch_name:
                continue
            _ch_einh = st.session_state.get(f'ch{_i}_einheit', 'µm')
            st.caption(f"{_ch_name}  ({_ch_einh})")
            _gc1, _gc2 = st.columns(2)
            _gc1.number_input(f"min {_ch_name}", step=1.0, format="%.2f",
                              key=f'ch{_i}_ymin', label_visibility="collapsed")
            _gc2.number_input(f"max {_ch_name}", step=1.0, format="%.2f",
                              key=f'ch{_i}_ymax', label_visibility="collapsed")
        st.caption("D (1. Ableitung)")
        _vc1, _vc2 = st.columns(2)
        _vc1.number_input("min (D)", step=100.0, format="%.0f",
                          key="v_axis_min", label_visibility="collapsed")
        _vc2.number_input("max (D)", step=100.0, format="%.0f",
                          key="v_axis_max", label_visibility="collapsed")
        st.caption("D2 (2. Ableitung)")
        _ac1, _ac2 = st.columns(2)
        _ac1.number_input("min (D2)", step=500.0, format="%.0f",
                          key="a_axis_min", label_visibility="collapsed")
        _ac2.number_input("max (D2)", step=500.0, format="%.0f",
                          key="a_axis_max", label_visibility="collapsed")

    with st.expander("Speichern / Laden", expanded=st.session_state.sub_speichern, key="sub_speichern", on_change=_SUB_EXPANDER_CBS['sub_speichern']):
        _json_str = json.dumps(
            {k: st.session_state.get(k) for k in EINSTELLUNGEN_KEYS},
            indent=2, ensure_ascii=False,
        )
        st.download_button(
            "💾 Einstellungen speichern",
            data=_json_str,
            file_name="einstellungen.json",
            mime="application/json",
            use_container_width=True,
            help="Alle aktuellen Einstellungen als JSON-Datei herunterladen.",
        )
        st.file_uploader(
            "Einstellungen laden", type=["json"],
            key="_settings_uploader",
            label_visibility="collapsed",
            help="JSON-Datei mit gespeicherten Einstellungen hochladen.",
            on_change=on_settings_upload,
        )
        _load_status = st.session_state.get('_settings_load_status')
        if _load_status == 'ok':
            st.success("Einstellungen geladen.")
            st.session_state['_settings_load_status'] = None
        elif _load_status is not None:
            st.error(f"Laden fehlgeschlagen: {_load_status}")

if sample_rate <= 0:
    st.sidebar.error("Samplerate muss größer als 0 sein.")
    st.stop()

if not uploaded_file:
    st.info("Bitte laden Sie eine CSV-Datei hoch, um die Analyse zu starten.")
    st.stop()

# ---------------------------------------------------------------------------
# KANAL-KONFIGURATION – aktive Kanäle aus Einstellungen ableiten
# ---------------------------------------------------------------------------

kanal_einheit_map = {
    nm: st.session_state.get(f'ch{i}_einheit', 'µm')
    for i, nm in enumerate(_kanal_cfg, 1)
    if nm
}

if len(kanal_namen_tuple) < 1:
    st.sidebar.error("Mindestens ein Kanalname muss angegeben sein.")
    st.stop()

# ---------------------------------------------------------------------------
# DATENAUFBEREITUNG
# ---------------------------------------------------------------------------

offs = tuple(st.session_state[f'off{i+1}'] for i in range(len(sensor_namen)))

_hz_faktor    = st.session_state.get('zeit_hz_faktor', 1000.0)
_zeit_einheit = _ZEIT_HZ_ZU_EINHEIT.get(round(_hz_faktor), 'ms')

df_full, sample_rate = _build_display_df_cached(
    df_raw, file_type, sample_rate, kanal_namen_tuple, offs,
    zeit_hz_faktor=_hz_faktor,
)

# ---------------------------------------------------------------------------
# X-OFFSET – als Sample-Shift einbauen
# RAW → [scale + Y-offset] = df_full  →  [X-shift] = df_use  →  [Crop] = df
# ---------------------------------------------------------------------------

_x_offs = [float(st.session_state.get(f'x_off{i+1}', 0.0)) for i in range(len(sensor_namen))]
if any(v != 0.0 for v in _x_offs) and len(df_full) > 1:
    _dt_ms = float(df_full['Zeit (ms)'].iloc[1] - df_full['Zeit (ms)'].iloc[0])
    df_use = df_full.copy()
    for _si, _sname in enumerate(sensor_namen):
        _n = int(round(_x_offs[_si] / _dt_ms)) if _dt_ms > 0 else 0
        if _n != 0:
            _col = np.roll(df_use[_sname].values, _n)
            if _n > 0:
                _col[:_n] = np.nan
            else:
                _col[_n:] = np.nan
            df_use[_sname] = _col
else:
    df_use = df_full

# ---------------------------------------------------------------------------
# AUTO-RESET BEI NEUER DATEI
# ---------------------------------------------------------------------------

if st.session_state.last_file_name != uploaded_file.name:
    total_time_ms = float(df_use['Zeit (ms)'].iloc[-1])
    for i, name in enumerate(sensor_namen, 1):
        off_init = float(df_raw[name].min()) * -1.0
        st.session_state[f'off{i}']        = off_init
        st.session_state[f'off{i}_slider'] = off_init
    for i in range(len(sensor_namen) + 1, N_KANÄLE + 1):
        st.session_state[f'off{i}']        = 0.0
        st.session_state[f'off{i}_slider'] = 0.0
    for i in range(1, N_KANÄLE + 1):
        st.session_state[f'x_off{i}'] = 0.0
    _setze_cursor_position(total_time_ms)
    st.session_state.crop_start     = None
    st.session_state.crop_end       = None
    st.session_state.zoom_token    += 1
    st.session_state.last_file_name = uploaded_file.name
    st.rerun()

max_zeit_full = float(df_use['Zeit (ms)'].iloc[-1])
max_idx_full  = len(df_use) - 1

# ---------------------------------------------------------------------------
# AUTO-RESET BEI NEUER DATEI – guard, falls Sidebar-Reset nicht greifen konnte
# ---------------------------------------------------------------------------

if st.session_state.last_file_name != uploaded_file.name:
    total_time_ms = float(df_use['Zeit (ms)'].iloc[-1])
    for i, name in enumerate(sensor_namen, 1):
        off_init = float(df_raw[name].min()) * -1.0
        st.session_state[f'off{i}']        = off_init
        st.session_state[f'off{i}_slider'] = off_init
    for i in range(len(sensor_namen) + 1, N_KANÄLE + 1):
        st.session_state[f'off{i}']        = 0.0
        st.session_state[f'off{i}_slider'] = 0.0
    for i in range(1, N_KANÄLE + 1):
        st.session_state[f'x_off{i}'] = 0.0
    _setze_cursor_position(total_time_ms)
    st.session_state.crop_start     = None
    st.session_state.crop_end       = None
    st.session_state.zoom_token    += 1
    st.session_state.last_file_name = uploaded_file.name
    st.rerun()

# ---------------------------------------------------------------------------
# CROP-LOGIK
# ---------------------------------------------------------------------------

crop_active = (
    st.session_state.crop_start is not None
    and st.session_state.crop_end is not None
)
if crop_active:
    ci_start = get_idx_at_x(st.session_state.crop_start, sample_rate, max_idx_full, _hz_faktor)
    ci_end   = get_idx_at_x(st.session_state.crop_end,   sample_rate, max_idx_full, _hz_faktor)
    df       = df_use.iloc[ci_start:ci_end + 1].reset_index(drop=True)
    min_zeit = float(df['Zeit (ms)'].iloc[0])
    max_zeit = float(df['Zeit (ms)'].iloc[-1])
    max_idx  = len(df) - 1
else:
    df       = df_use
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
    help="Aktiver Kanal für alle Berechnungen: Cursor-Messung, D-max, D2-max und SOP.",
)

st.sidebar.caption("Anzeige", help="Kanäle ein-/ausblenden. Der aktive Mess-Kanal ist immer sichtbar.")
_anz_cols = st.sidebar.columns(3)
for _ai, _aname in enumerate(sensor_namen):
    _ch_n = _sensor_ch_num[_aname]
    _anz_cols[_ai % 3].checkbox(
        _aname,
        key=f'show_ch{_ch_n}',
        disabled=(_aname == active_sensor),
    )

sichtbare_sensor_namen = [
    name for name in sensor_namen
    if st.session_state.get(f'show_ch{_sensor_ch_num[name]}', True) or name == active_sensor
]

_aktiv_einheit = kanal_einheit_map.get(active_sensor, 'µm')
v_einheit, a_einheit, v_faktor, a_faktor = _ableit_info(_aktiv_einheit, _zeit_einheit)

xa = float(np.clip(st.session_state.xa, min_zeit, max_zeit))
xb = float(np.clip(st.session_state.xb, min_zeit, max_zeit))
if (xa != st.session_state.xa
        or xa != st.session_state.get('xa_sw', xa)
        or xa != st.session_state.get('xa_nw', xa)):
    st.session_state.xa    = xa
    st.session_state.xa_sw = xa
    st.session_state.xa_nw = xa
if (xb != st.session_state.xb
        or xb != st.session_state.get('xb_sw', xb)
        or xb != st.session_state.get('xb_nw', xb)):
    st.session_state.xb    = xb
    st.session_state.xb_sw = xb
    st.session_state.xb_nw = xb

with st.sidebar.expander("Zeitmarker & Basis", expanded=False):
    st.number_input(
        f"Zeit XA ({_zeit_einheit})", min_zeit, max_zeit,
        step=0.001, format="%.3f",
        key="xa_nw", on_change=update_xa_from_num,
        help=f"Linker Zeitcursor ({_zeit_einheit}) – Startpunkt für Δt, Δs und D (A-B).",
    )
    st.number_input(
        f"Zeit XB ({_zeit_einheit})", min_zeit, max_zeit,
        step=0.001, format="%.3f",
        key="xb_nw", on_change=update_xb_from_num,
        help=f"Rechter Zeitcursor ({_zeit_einheit}) – Endpunkt für Δt, Δs und D (A-B).",
    )
    if xa > xb:
        st.warning("⚠️ XA liegt nach XB – Marker vertauscht.")
    _tb_f = _hz_faktor / 1e3   # ms → aktuelle Zeiteinheit (Rückrechnung)
    _x_len = max(1e-9, max_zeit - min_zeit)
    _tb_min_curr = _x_len / 200
    _tb_max_curr = _x_len * 0.75
    _val_s_min = _tb_min_curr * _ZEIT_TO_S.get(_zeit_einheit, 1e-3)
    _tb_s_einheit = 'ns'
    for _u in ('s', 'ms', 'µs', 'ns'):
        if abs(_val_s_min) / _ZEIT_TO_S[_u] >= 0.001:
            _tb_s_einheit = _u
            break
    _tb_s_scale = _ZEIT_TO_S.get(_zeit_einheit, 1e-3) / _ZEIT_TO_S[_tb_s_einheit]
    _tb_min_d = _tb_min_curr * _tb_s_scale
    _tb_max_d = _tb_max_curr * _tb_s_scale
    _tb_def_d = float(np.clip(_x_len * 0.03 * _tb_s_scale, _tb_min_d, _tb_max_d))
    _tb_step_d = max(1e-12, (_tb_max_d - _tb_min_d) / 100)
    _tb_mag = _tb_min_d
    _tb_fmt = ("%.0f" if _tb_mag >= 10 else
               "%.2f" if _tb_mag >= 0.1 else
               "%.4f" if _tb_mag >= 0.001 else "%.6f")
    _v_tb_display = st.slider(
        f"Zeitbasis D-max ({_tb_s_einheit})",
        _tb_min_d, _tb_max_d, _tb_def_d,
        step=_tb_step_d, format=f"{_tb_fmt} {_tb_s_einheit}",
        help="Mittelungsfenster für D-max, D2-max und SOP: Der Peak wird über dieses Zeitfenster gemittelt. Kleiner = empfindlicher, größer = robuster gegenüber Rauschen.",
    )
    v_time_base_ms = (_v_tb_display / _tb_s_scale) / _tb_f   # → ms

show_v_avg    = st.sidebar.toggle("Schnittlinie A–B anzeigen", key="show_v_avg",
                                  help="Zeichnet eine Verbindungslinie von XA nach XB und visualisiert damit die mittlere Änderungsrate D (A-B).")
show_rect_fit = st.sidebar.toggle(
    "Rechteck-Fit füllen", key="show_rect_fit",
    help="Zeigt zusätzlich vertikale Kantenlinien und hellgrüne Füllung für alle erkannten Rechteck-Pulse.",
)
show_integral = st.sidebar.toggle(
    "I anzeigen (Integration)", key="show_integral",
    help="Zeichnet die Fläche unter der aktiven Kurve zwischen XA und XB transparent ein.",
)
show_velocity = st.sidebar.toggle(
    "D anzeigen (1. Ableitung)", key="show_velocity",
    help="Zeigt die 1. Ableitung des aktiven Kanals auf einer zweiten Y-Achse rechts.",
)
if show_velocity:
    st.sidebar.slider(
        "Glättung D", 5, 80, step=1,
        key="window_length",
        help="Fenstergröße des Savitzky-Golay-Filters für die 1. Ableitung. Größer = glatter, aber geringere Detailauflösung.",
    )
show_acceleration = st.sidebar.toggle(
    "D2 anzeigen (2. Ableitung)", key="show_acceleration",
    help="Zeigt die 2. Ableitung des aktiven Kanals auf einer dritten Y-Achse rechts.",
)
if show_acceleration:
    st.sidebar.slider(
        "Glättung D2", 10, 75, step=1,
        key="window_length_accel",
        help="Fenstergröße des Savitzky-Golay-Filters für die 2. Ableitung. Größere Werte nötig, da die 2. Ableitung stärker rauscht.",
    )

show_sop = st.sidebar.toggle(
    "Speed on Point (SOP)", key="show_sop",
    help="Misst die Geschwindigkeit an der steigenden Flanke des Rechtecksignals auf einem einstellbaren Hub-Pegel. Erfordert erkanntes Rechteck-Fit.",
)
if show_sop:
    st.sidebar.slider(
        "SOP Pegel (%)", 0, 100, step=1,
        key="sop_percent",
        help="Höhe auf der steigenden Flanke in Prozent des Hub (0 % = unterer Pegel, 100 % = oberer Pegel).",
    )

rect_fit = compute_best_fit_rectangle(
    df_use['Zeit (ms)'].values,
    df_use[active_sensor].values,
)

# ---------------------------------------------------------------------------
# MESSWERTBERECHNUNG
# ---------------------------------------------------------------------------

if crop_active:
    idx_a = get_idx_at_x(xa - min_zeit, sample_rate, max_idx, _hz_faktor)
    idx_b = get_idx_at_x(xb - min_zeit, sample_rate, max_idx, _hz_faktor)
else:
    idx_a = get_idx_at_x(xa, sample_rate, max_idx, _hz_faktor)
    idx_b = get_idx_at_x(xb, sample_rate, max_idx, _hz_faktor)

ya = df.loc[idx_a, active_sensor]
yb = df.loc[idx_b, active_sensor]

dt_val_ms = abs(xb - xa)
dy        = abs(yb - ya)
v_avg     = dy / dt_val_ms * _hz_faktor * v_faktor if dt_val_ms > 0 else 0.0

halbes_zeitfenster = max(1, int(v_time_base_ms / 1000.0 * sample_rate / 2))

def v_at_cursor(idx: int) -> float:
    """Mittlere D um idx herum (in Anzeigeeinheit)."""
    i0 = max(0, idx - halbes_zeitfenster)
    i1 = min(max_idx, idx + halbes_zeitfenster)
    if i1 <= i0:
        return float('nan')
    _dy  = float(df.loc[i1, active_sensor] - df.loc[i0, active_sensor])
    dt_s = (i1 - i0) / sample_rate
    return (_dy * v_faktor) / dt_s

v_at_xa        = v_at_cursor(idx_a)
v_at_xb        = v_at_cursor(idx_b)
v_cursor_delta = (
    abs(v_at_xb - v_at_xa)
    if not (np.isnan(v_at_xa) or np.isnan(v_at_xb))
    else float('nan')
)

idx_start, idx_end = sorted([idx_a, idx_b])

# Integral unter der aktiven Kurve zwischen XA und XB (Trapezregel, volle Auflösung)
if idx_end > idx_start:
    _t_int = df.loc[idx_start:idx_end, 'Zeit (ms)'].values
    _y_int = df.loc[idx_start:idx_end, active_sensor].values
    integral_val = float(np.trapezoid(_y_int, _t_int))   # [aktiv_einheit × zeit_einheit]
else:
    integral_val = 0.0

# ---------------------------------------------------------------------------
# SG-ABLEITUNGEN – einmalig auf vollem Datensatz (Messwerte + Diagramm)
# ---------------------------------------------------------------------------

arr_full  = df[active_sensor].values
dt_step_s = 1.0 / sample_rate
sg_v_roh  = reader.berechne_sg_ableitung(arr_full, dt_step_s, st.session_state.window_length, 1)
sg_a_roh  = reader.berechne_sg_ableitung(arr_full, dt_step_s, st.session_state.window_length_accel, 2)

# Peak-Marker-Initialisierung (werden nur gesetzt wenn genug Datenpunkte vorhanden)
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
    if sg_v_roh is not None:
        sg_v_abs  = np.abs(sg_v_roh * v_faktor)
        sg_v_slice        = sg_v_abs[idx_start:idx_end + 1]
        idx_vmax_peak_loc = int(np.argmax(sg_v_slice))
        idx_vmax_peak     = idx_start + idx_vmax_peak_loc
        iv_start          = max(0, idx_vmax_peak - halbes_zeitfenster)
        iv_ende           = min(max_idx, idx_vmax_peak + halbes_zeitfenster)
        v_max             = float(np.mean(sg_v_abs[iv_start:iv_ende + 1]))

        if 0 <= iv_start <= max_idx and 0 <= iv_ende <= max_idx:
            t_vmax_start = df.loc[iv_start, 'Zeit (ms)']
            y_vmax_start = df.loc[iv_start, active_sensor]
            t_vmax_ende  = df.loc[iv_ende,  'Zeit (ms)']
            y_vmax_ende  = df.loc[iv_ende,  active_sensor]
            has_vmax     = True

    if sg_a_roh is not None:
        sg_a = sg_a_roh * a_faktor

        def _peak_marker(idx_abs):
            """Gemittelter Beschleunigungswert und Diagramm-Position für einen Peak."""
            ia0  = max(0, idx_abs - halbes_zeitfenster)
            ia1  = min(max_idx, idx_abs + halbes_zeitfenster)
            wert = float(np.mean(sg_a[ia0:ia1 + 1]))
            return wert, float(df.loc[idx_abs, 'Zeit (ms)']), float(df.loc[idx_abs, active_sensor])

        sg_a_slice = sg_a[idx_start:idx_end + 1]

        idx_falling_abs                                = idx_start + int(np.argmax(sg_a_slice))
        a_max_falling, t_amax_falling, y_amax_falling = _peak_marker(idx_falling_abs)
        has_amax_falling = True

        idx_rising_abs                               = idx_start + int(np.argmin(sg_a_slice))
        a_min_rising, t_amax_rising, y_amax_rising   = _peak_marker(idx_rising_abs)
        has_amax_rising = True

if show_sop and rect_fit is not None:
    sop_linien, v_sop = _finde_sop_kreuzungen(
        df_use['Zeit (ms)'].values,
        df_use[active_sensor].values,
        rect_fit,
        st.session_state.sop_percent,
        sample_rate,
        halbes_zeitfenster,
        v_faktor=v_faktor,
    )

# ---------------------------------------------------------------------------
# DOWNSAMPLING FÜR GROSSE DATEIEN
# ---------------------------------------------------------------------------

if len(df) > MAX_PLOT_PUNKTE:
    _ds_step = len(df) // MAX_PLOT_PUNKTE
    df_plot  = df.iloc[::_ds_step]
else:
    _ds_step = 1
    df_plot  = df

# Ableitungen für Diagramm: SG-Arrays (volle Länge) mit gleichem Schritt slicen –
# spart einen kompletten SG-Durchlauf gegenüber Neuberechnung auf df_plot.
velocity     = (sg_v_roh * v_faktor)[::_ds_step] if show_velocity     and sg_v_roh is not None else None
acceleration = (sg_a_roh * a_faktor)[::_ds_step] if show_acceleration and sg_a_roh is not None else None

# ---------------------------------------------------------------------------
# DIAGRAMM AUFBAUEN
# ---------------------------------------------------------------------------

# Y-Bereiche aus vollständigen Daten (df_use) – Crop darf die Y-Achse nicht verschieben
_prim_einheit = next(iter(dict.fromkeys(kanal_einheit_map.get(n, 'µm') for n in sichtbare_sensor_namen)), 'µm')
_prim_namen   = [n for n in sichtbare_sensor_namen if kanal_einheit_map.get(n, 'µm') == _prim_einheit] or sichtbare_sensor_namen

_prim_namen_full = [n for n in _prim_namen if n in df_use.columns]
_y_full_prim = df_use[_prim_namen_full] if _prim_namen_full else None
if _y_full_prim is not None and not _y_full_prim.empty:
    y_max_plot = float(_y_full_prim.max().max())
    y_min_plot = float(_y_full_prim.min().min())
else:
    y_max_plot = float(df_plot[_prim_namen].max().max())
    y_min_plot = float(df_plot[_prim_namen].min().min())
y_range_plot = [y_min_plot, y_max_plot + (y_max_plot - y_min_plot) * Y_PUFFER]

_yrange_fallback: dict[str, list] = {}
for _e in set(kanal_einheit_map.get(n, 'µm') for n in sichtbare_sensor_namen):
    _cols_e = [n for n in sichtbare_sensor_namen if kanal_einheit_map.get(n, 'µm') == _e and n in df_use.columns]
    if _cols_e:
        _lo_e   = float(df_use[_cols_e].min().min())
        _hi_e   = float(df_use[_cols_e].max().max())
        _span_e = _hi_e - _lo_e
        _yrange_fallback[_e] = [_lo_e, _hi_e + _span_e * Y_PUFFER]

_kanal_bereiche: dict[str, tuple[float, float]] = {}
for _n in sichtbare_sensor_namen:
    if _n in df_use.columns:
        _kanal_bereiche[_n] = (float(df_use[_n].min()), float(df_use[_n].max()))

velocity_ok     = velocity is not None
acceleration_ok = acceleration is not None
_kanal_farbe_map = {name: KANAL_FARBEN[sensor_namen.index(name)] for name in sichtbare_sensor_namen}
kanal_zu_yaxis, layout_yachsen, v_yaxis, a_yaxis, x_domain_end = _yachsen_layout(
    sichtbare_sensor_namen, kanal_einheit_map, y_range_plot,
    show_velocity, velocity_ok, show_acceleration, acceleration_ok,
    v_einheit=v_einheit, a_einheit=a_einheit,
    kanal_farbe_map=_kanal_farbe_map,
    y_ranges_fallback=_yrange_fallback,
    kanal_bereiche=_kanal_bereiche,
    kanal_ch_num=_sensor_ch_num,
)
active_yaxis = kanal_zu_yaxis.get(active_sensor, 'y')

fig = go.Figure()

_baue_traces(
    fig, df_plot, sichtbare_sensor_namen,
    kanal_zu_yaxis, active_sensor, active_yaxis,
    sensor_namen,
    xa, xb, ya, yb, min_zeit, max_zeit,
    show_v_avg, rect_fit, show_rect_fit,
    has_vmax, t_vmax_start, y_vmax_start, t_vmax_ende, y_vmax_ende,
    has_amax_falling, t_amax_falling, y_amax_falling,
    has_amax_rising, t_amax_rising, y_amax_rising,
    sop_linien,
    show_velocity, velocity, v_yaxis,
    show_acceleration, acceleration, a_yaxis,
    show_integral,
)

_y_lim_keys = (
    ['v_axis_min', 'v_axis_max', 'a_axis_min', 'a_axis_max']
    + [f'ch{_i}_ymin' for _i in range(1, N_KANÄLE + 1)]
    + [f'ch{_i}_ymax' for _i in range(1, N_KANÄLE + 1)]
)
_y_lim_token = hash(tuple(st.session_state.get(k, 0) for k in _y_lim_keys))
_axis_token  = hash(tuple(sorted(kanal_zu_yaxis.items())))

fig.update_layout(
    xaxis_title=f"Zeit ({_zeit_einheit})",
    height=600,
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
    uirevision=f"{st.session_state.zoom_token}-{st.session_state.crop_start}-{st.session_state.crop_end}-{_y_lim_token}-{_axis_token}-{round(max_zeit_full, 4)}",
    xaxis=dict(autorange=True, rangemode='nonnegative', domain=[0, x_domain_end],
               showgrid=True, gridcolor='rgba(180,180,180,0.4)', gridwidth=1, nticks=20),
    **layout_yachsen,
)
st.plotly_chart(fig, width="stretch", key="main_chart")

# ---------------------------------------------------------------------------
# CURSOR-SLIDER
# Linke Spalte (0.04): gleicht Y-Achsen-Breite links aus
# Rechte Spalte (1 - x_domain_end): gleicht rechte Y-Achsen aus
# ---------------------------------------------------------------------------

_slider_right_pad = round(1.0 - x_domain_end, 4)
if _slider_right_pad > 0.01:
    c_pad, c_slider, _ = st.columns([0.04, x_domain_end - 0.04, _slider_right_pad])
else:
    c_pad, c_slider = st.columns([0.04, 0.96])
with c_slider:
    st.slider(
        "XA", min_zeit, max_zeit,
        key="xa_sw", step=0.001, format=f"%.3f {_zeit_einheit}",
        on_change=update_xa_from_slider, label_visibility="collapsed",
        help=f"Linker Cursor XA ({_zeit_einheit}) – ziehen oder Wert im Expander 'Zeitmarker & Basis' eingeben.",
    )
    st.slider(
        "XB", min_zeit, max_zeit,
        key="xb_sw", step=0.001, format=f"%.3f {_zeit_einheit}",
        on_change=update_xb_from_slider, label_visibility="collapsed",
        help=f"Rechter Cursor XB ({_zeit_einheit}) – ziehen oder Wert im Expander 'Zeitmarker & Basis' eingeben.",
    )

# ---------------------------------------------------------------------------
# CROP / SHOW ALL
# ---------------------------------------------------------------------------

margin  = abs(xb - xa) * CROP_RAND_FAKTOR
crop_t0 = max(min_zeit, min(xa, xb) - margin)
crop_t1 = min(max_zeit, max(xa, xb) + margin)

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    if st.button("✂️ Crop A–B  (+15%)", disabled=(dt_val_ms == 0), width="stretch",
                 help="Schneidet die Ansicht auf den Bereich zwischen XA und XB zu (je 15 % Rand beiderseits)."):
        st.session_state.crop_start = crop_t0
        st.session_state.crop_end   = crop_t1
        st.session_state.xa    = float(min(xa, xb))
        st.session_state.xb    = float(max(xa, xb))
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
        f"✂️ Crop aktiv: {st.session_state.crop_start:.3f} {_zeit_einheit} – "
        f"{st.session_state.crop_end:.3f} {_zeit_einheit}"
    )

# ---------------------------------------------------------------------------
# KENNGRÖSSEN-ANZEIGE
# ---------------------------------------------------------------------------

freq_hz = (_hz_faktor / dt_val_ms) if dt_val_ms > 0 else float('nan')
hub     = abs(rect_fit['y_high'] - rect_fit['y_low']) if rect_fit is not None else float('nan')

# Zeile 1 – Zeit & Signal
z1, z2, z3, z4 = st.columns(4)
z1.metric("Δt (A-B)",          _fmt_zeit(dt_val_ms, _zeit_einheit))
z2.metric("Frequenz Δt (A-B)", _fmt_freq(freq_hz))
z3.metric("Δs (A-B)",          _fmt_val(dy, _aktiv_einheit))
z4.metric("Hub Best-fit",      _fmt_val(hub, _aktiv_einheit) if not np.isnan(hub) else "N/A")

# Zeile 2 – D (1. Ableitung)
g1, g2, g3, g4 = st.columns(4)
g1.metric("D (A-B)",           _fmt_val(v_avg, v_einheit))
g2.metric("ΔD (A-B)",          _fmt_val(v_cursor_delta, v_einheit) if not np.isnan(v_cursor_delta) else "N/A")
g3.metric("D max (Peak)",      _fmt_val(v_max, v_einheit)          if not np.isnan(v_max) else "N/A")
g4.metric("SOP",               _fmt_val(v_sop, v_einheit)          if not np.isnan(v_sop) else "N/A")

# Zeile 3 – D2 (2. Ableitung) + Integral
a1, a2, a3 = st.columns(3)
a1.metric("D2 max Fall.",      _fmt_val(a_max_falling, a_einheit)  if not np.isnan(a_max_falling) else "N/A")
a2.metric("D2 min Rise.",      _fmt_val(a_min_rising, a_einheit)   if not np.isnan(a_min_rising) else "N/A")
a3.metric("∫ (A-B)",           _fmt_integral(integral_val, _aktiv_einheit, _zeit_einheit))

# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------

st.sidebar.header("3. Export")
metrics = {
    # Zeit & Signal
    f"XA ({_zeit_einheit})":    _fmt_zeit(xa, _zeit_einheit),
    f"XB ({_zeit_einheit})":    _fmt_zeit(xb, _zeit_einheit),
    "Δt (A-B)":                 _fmt_zeit(dt_val_ms, _zeit_einheit),
    "Frequenz Δt (A-B)":        _fmt_freq(freq_hz),
    "Δs (A-B)":                 _fmt_val(dy, _aktiv_einheit),
    "Hub Best-fit":             _fmt_val(hub, _aktiv_einheit)              if not np.isnan(hub) else "N/A",
    # D (1. Ableitung)
    "D (A-B)":                  _fmt_val(v_avg, v_einheit),
    "ΔD (A-B)":                 _fmt_val(v_cursor_delta, v_einheit)        if not np.isnan(v_cursor_delta) else "N/A",
    "D max (Peak)":             _fmt_val(v_max, v_einheit)                 if not np.isnan(v_max) else "N/A",
    "SOP":                      _fmt_val(v_sop, v_einheit)                 if not np.isnan(v_sop) else "N/A",
    # D2 (2. Ableitung)
    "D2 max Fall.":             _fmt_val(a_max_falling, a_einheit)         if not np.isnan(a_max_falling) else "N/A",
    "D2 min Rise.":             _fmt_val(a_min_rising, a_einheit)          if not np.isnan(a_min_rising) else "N/A",
    # Integral
    "∫ (A-B)":                  _fmt_integral(integral_val, _aktiv_einheit, _zeit_einheit),
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
                df, sichtbare_sensor_namen, active_sensor,
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
                kanal_einheit_map=kanal_einheit_map,
                alle_sensor_namen=sensor_namen,
                hz_faktor=_hz_faktor,
                zeit_einheit=_zeit_einheit,
                kanal_ch_num=_sensor_ch_num,
                show_integral=show_integral,
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
