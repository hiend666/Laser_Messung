# pip install streamlit pandas plotly scipy "kaleido==0.2.1" reportlab
# streamlit run app.py
"""
Messdaten-Auswertung – Laservibrometer CSV.

Zeigt Signal-, D- und D2-Kurven (1. und 2. Ableitung) und
exportiert Ergebnisse als PDF oder PNG.
"""
import io
import json
import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

import reader

VERSION = "v1.00.14"

# ---------------------------------------------------------------------------
# KONSTANTEN
# ---------------------------------------------------------------------------

MAX_PLOT_PUNKTE = 5_000     # Downsampling-Schwelle für interaktives Diagramm

# Einheiten-Auswahl für Kanäle
EINHEIT_OPTIONEN = ['µm', 'mm', 'm', 'V', 'mV', 'A', 'mA', 'N', 'kN', 'bar', 'Pa', '°C', '%']
EINHEIT_ALLE = ['µm', 'mm', 'm', 'V', 'mV', 'A', 'mA', 'N', 'kN', 'bar', 'Pa', '°C', '%']

N_KANÄLE    = 4     # Maximale Kanalanzahl – einzige Stelle um diese zu ändern
SPLIT_FAKTOR = 15.0  # Y-Achsen-Aufteilung bei gleicher Einheit wenn Bereiche > Faktor abweichen

def _einh_sfx(einheit: str) -> str:
    return einheit.replace('µ', 'u').replace('°', 'deg').replace('%', 'pct').replace('/', 'p')

def einheit_ss_key_min(einheit: str) -> str:
    return 'ymin_' + _einh_sfx(einheit)

def einheit_ss_key_max(einheit: str) -> str:
    return 'ymax_' + _einh_sfx(einheit)

# Diagramm-Farben – Kanäle
FARBE_KANAL1    = '#003366'
FARBE_KANAL2    = '#4c78a8'
FARBE_KANAL3    = '#d62728'
FARBE_KANAL4    = '#2ca02c'
KANAL_FARBEN    = [FARBE_KANAL1, FARBE_KANAL2, FARBE_KANAL3, FARBE_KANAL4]

# Diagramm-Farben – Auswertung
FARBE_D         = 'purple'
FARBE_D2        = 'orange'
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
# Kanal-Standardnamen und Oszilloskop-Skalierungen (Index 0 = Kanal 1)
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
    'v_axis_min': -3_200,
    'v_axis_max':  3_200,
    'a_axis_min': -20_000,
    'a_axis_max':  20_000,
    'zeit_hz_faktor': 1000.0,  # Umrechnungsfaktor s → Anzeigeeinheit (1e3=ms, 1e6=µs, 1e9=ns)
    'n_kanäle_datei': N_KANÄLE,  # Kanalanzahl laut geladener Datei
    'sub_dateityp':   True,
    'sub_einlesen':   False,
    'sub_kanaele':    False,
    'sub_offsets':    False,
    'sub_xoffset':    False,
    'sub_grenzwerte': False,
    'sub_speichern':  False,
    'einstellungen':  True,
}
# Pro-Kanal-Defaults per Loop – einzige Stelle mit kanalanzahlabhängiger Logik
for _i in range(1, N_KANÄLE + 1):
    defaults[f'ch{_i}_name']    = _CH_NAMEN_DEFAULT[_i - 1]
    defaults[f'ch{_i}_einheit'] = 'µm'
    defaults[f'osc_skale_{_i}'] = _OSC_SKALE_DEFAULT[_i - 1]
    defaults[f'off{_i}']        = 0.0
    defaults[f'off{_i}_slider'] = 0.0
    defaults[f'x_off{_i}']     = 0.0
    defaults[f'show_ch{_i}']   = True
# Y-Achsen-Grenzwerte pro Einheit (0 = automatisch)
for _e in EINHEIT_ALLE:
    defaults[einheit_ss_key_min(_e)] = 0.0
    defaults[einheit_ss_key_max(_e)] = 0.0

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Keys die beim JSON-Export/Import gespeichert werden
EINSTELLUNGEN_KEYS: list[str] = [
    'file_type_radio',
    'sample_rate', 'sample_rate_unit', 'sample_rate_unit_toggle',
    'skip_rows', 'max_samples',
    'xa', 'xb',
    'show_v_avg', 'show_rect_fit',
    'show_velocity', 'window_length',
    'show_acceleration', 'window_length_accel',
    'show_sop', 'sop_percent',
    'v_axis_min', 'v_axis_max', 'a_axis_min', 'a_axis_max',
]
for _i in range(1, N_KANÄLE + 1):
    EINSTELLUNGEN_KEYS += [
        f'ch{_i}_name', f'ch{_i}_einheit', f'osc_skale_{_i}',
        f'off{_i}', f'off{_i}_slider', f'x_off{_i}', f'show_ch{_i}',
    ]
for _e in EINHEIT_ALLE:
    EINSTELLUNGEN_KEYS.append(einheit_ss_key_min(_e))
    EINSTELLUNGEN_KEYS.append(einheit_ss_key_max(_e))


# ---------------------------------------------------------------------------
# GECACHTE DATENFUNKTIONEN  (Wrapper um reader.py – Streamlit-Caching hier)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# HILFSFUNKTION – MULTI-ACHSEN-LAYOUT
# ---------------------------------------------------------------------------

def _yachsen_layout(
    kanal_namen: list[str],
    kanal_einheit_map: dict[str, str],
    y_range_primaer,
    show_velocity: bool,
    velocity_ok: bool,
    show_acceleration: bool,
    acceleration_ok: bool,
    v_einheit: str = 'mm/s',
    a_einheit: str = 'm/s²',
    kanal_farbe_map: dict[str, str] | None = None,
    y_ranges_fallback: dict[str, list] | None = None,
    kanal_bereiche: dict[str, tuple[float, float]] | None = None,
) -> tuple[dict, dict, str, str, float]:
    """Berechnet Y-Achsen-Zuordnung und Plotly-Layout für alle Achsen.

    Kanäle gleicher Einheit teilen eine Achse – außer wenn ihre Wertebereiche
    um mehr als SPLIT_FAKTOR voneinander abweichen (dann separate Achsen).
    Achsen mit identischen manuellen Grenzen werden zusammengeführt;
    der Achstitel zeigt alle betroffenen Einheiten (z.B. 'µm / V').
    Die erste Einheit landet links, alle weiteren rechts.

    Gibt zurück:
    - kanal_zu_yaxis: {'Festo': 'y', 'DST': 'y2', ...}
    - layout_yachsen:  dict für fig.update_layout(**layout_yachsen)
    - v_yaxis:         yaxis-String für Geschwindigkeit
    - a_yaxis:         yaxis-String für Beschleunigung
    - x_domain_end:    rechte Grenze des Plot-Bereichs (0…1)
    """
    STEP = 0.07

    def _user_lim(einheit: str) -> tuple[float, float] | None:
        lo = float(st.session_state.get(einheit_ss_key_min(einheit), 0))
        hi = float(st.session_state.get(einheit_ss_key_max(einheit), 0))
        return (lo, hi) if (lo != 0 or hi != 0) else None

    def _kanal_span(name: str) -> float | None:
        if kanal_bereiche and name in kanal_bereiche:
            lo, hi = kanal_bereiche[name]
            return abs(hi - lo)
        return None

    def _achsfarbe(kanäle: list[str]) -> dict:
        if not kanal_farbe_map:
            return {}
        for n in kanäle:
            if n in kanal_farbe_map:
                return dict(color=kanal_farbe_map[n])
        return {}

    def _fallback_rng(titel: str, kanäle: list[str]) -> list | None:
        if len(kanäle) == 1 and kanal_bereiche and kanäle[0] in kanal_bereiche:
            lo, hi = kanal_bereiche[kanäle[0]]
            span = abs(hi - lo)
            return [lo, hi + span * 0.15]
        if y_ranges_fallback:
            for e in [e.strip() for e in titel.split(' / ')]:
                if e in y_ranges_fallback:
                    return y_ranges_fallback[e]
        return None

    def _rng(titel: str, kanäle: list[str]) -> list | None:
        for e in [e.strip() for e in titel.split(' / ')]:
            lo = float(st.session_state.get(einheit_ss_key_min(e), 0))
            hi = float(st.session_state.get(einheit_ss_key_max(e), 0))
            if lo != 0 or hi != 0:
                return [lo, hi]
        return _fallback_rng(titel, kanäle)

    # --- Schritt 1: Einheiten-basierte Gruppen (Reihenfolge aus kanal_namen) ---
    einheit_gruppen: dict[str, list[str]] = {}
    for n in kanal_namen:
        einheit_gruppen.setdefault(kanal_einheit_map.get(n, 'µm'), []).append(n)

    # --- Schritt 2: Aufteilen wenn Wertebereich > SPLIT_FAKTOR ---
    final_achsen: list[tuple[str, list[str]]] = []   # (titel, kanal_liste)
    for einheit, kanäle in einheit_gruppen.items():
        if len(kanäle) <= 1 or _user_lim(einheit) is not None:
            final_achsen.append((einheit, list(kanäle)))
            continue
        spans = []
        for n in kanäle:
            s = _kanal_span(n)
            if s is not None and s > 0:
                spans.append(s)
        if len(spans) >= 2 and max(spans) / min(spans) > SPLIT_FAKTOR:
            for n in kanäle:
                final_achsen.append((einheit, [n]))
        else:
            final_achsen.append((einheit, list(kanäle)))

    # --- Schritt 3: Achsen mit gleichen manuellen Grenzen zusammenführen ---
    merged_achsen: list[tuple[str, list[str]]] = []
    lim_zu_idx: dict[tuple[float, float], int] = {}

    for titel, kanäle in final_achsen:
        lim = _user_lim(titel.split(' / ')[0].strip())
        if lim is not None:
            if lim in lim_zu_idx:
                idx = lim_zu_idx[lim]
                ex_titel, ex_kanäle = merged_achsen[idx]
                ex_einheiten = [e.strip() for e in ex_titel.split(' / ')]
                if titel not in ex_einheiten:
                    ex_titel = ex_titel + ' / ' + titel
                merged_achsen[idx] = (ex_titel, ex_kanäle + kanäle)
            else:
                lim_zu_idx[lim] = len(merged_achsen)
                merged_achsen.append((titel, kanäle))
        else:
            merged_achsen.append((titel, kanäle))

    final_achsen = merged_achsen

    # --- Schritt 4: Kanal → yaxis Zuordnung ---
    kanal_zu_yaxis: dict[str, str] = {}
    for i, (_, kanäle) in enumerate(final_achsen):
        ys = 'y' if i == 0 else f'y{i + 1}'
        for n in kanäle:
            kanal_zu_yaxis[n] = ys
    for n in kanal_namen:
        if n not in kanal_zu_yaxis:
            kanal_zu_yaxis[n] = 'y'

    n_sig = len(final_achsen)
    v_yaxis = f'y{n_sig + 1}'
    a_yaxis = f'y{n_sig + 2}'

    # Rechte Signal-Achsen
    rechte_achsen: list[tuple[str, str, list | None, dict]] = []
    for i, (titel, kanäle) in enumerate(final_achsen[1:], 1):
        rechte_achsen.append((f'yaxis{i + 1}', titel, _rng(titel, kanäle), _achsfarbe(kanäle)))
    if show_velocity and velocity_ok:
        v_lo = float(st.session_state.get('v_axis_min', 0))
        v_hi = float(st.session_state.get('v_axis_max', 0))
        v_rng = [v_lo, v_hi] if not (v_lo == 0 and v_hi == 0) else None
        rechte_achsen.append((f'yaxis{n_sig + 1}', f'D ({v_einheit})', v_rng, dict(color=FARBE_D)))
    if show_acceleration and acceleration_ok:
        a_lo = float(st.session_state.get('a_axis_min', 0))
        a_hi = float(st.session_state.get('a_axis_max', 0))
        a_rng = [a_lo, a_hi] if not (a_lo == 0 and a_hi == 0) else None
        rechte_achsen.append((f'yaxis{n_sig + 2}', f'D2 ({a_einheit})', a_rng, dict(color=FARBE_D2)))

    n_right = len(rechte_achsen)
    x_domain_end = max(0.5, 1.0 - STEP * max(0, n_right - 1)) if n_right > 1 else 1.0

    # Primäre linke Achse
    titel0, kanäle0 = final_achsen[0] if final_achsen else ('µm', [])
    rng0 = _rng(titel0, kanäle0) if final_achsen else None
    layout_yachsen: dict = {
        'yaxis': dict(title=titel0, range=rng0 if rng0 else y_range_primaer,
                      **_achsfarbe(kanäle0))
    }

    for idx, (yk, title, rng, farb_dict) in enumerate(rechte_achsen):
        pos = (x_domain_end + STEP * idx) if n_right > 1 else 1.0
        ax: dict = dict(title=title, overlaying='y', side='right', showgrid=False,
                        position=pos, anchor='free', **farb_dict)
        if rng:
            ax['range'] = rng
        layout_yachsen[yk] = ax

    return kanal_zu_yaxis, layout_yachsen, v_yaxis, a_yaxis, x_domain_end


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
    val = max(0.0, float(st.session_state.xa_sw))
    st.session_state.xa    = val
    st.session_state.xa_nw = val   # Zahlenfeld synchron halten

def update_xa_from_num():
    val = max(0.0, float(st.session_state.xa_nw))
    st.session_state.xa    = val
    st.session_state.xa_sw = val   # Slider synchron halten

def update_xb_from_slider():
    val = max(0.0, float(st.session_state.xb_sw))
    st.session_state.xb    = val
    st.session_state.xb_nw = val

def update_xb_from_num():
    val = max(0.0, float(st.session_state.xb_nw))
    st.session_state.xb    = val
    st.session_state.xb_sw = val

def _make_off_cb(i: int):
    def _cb(): st.session_state[f'off{i}'] = st.session_state[f'off{i}_slider']
    return _cb

# Indizierter Zugriff via OFF_CALLBACKS[kanal_index_0basiert]
OFF_CALLBACKS = [_make_off_cb(i) for i in range(1, N_KANÄLE + 1)]


def _ableit_info(einheit: str) -> tuple[str, str, float, float]:
    """(v_einheit, a_einheit, v_faktor, a_faktor) für eine Kanal-Einheit.

    Für µm: Umrechnung in konventionelle Einheiten (mm/s, m/s²).
    Für alle anderen Einheiten: einheit/s und einheit/s² ohne Umrechnung.
    v_faktor/a_faktor wandeln SG-Rohableitung ([einheit]/s, [einheit]/s²) in
    die Anzeigeeinheit um.
    """
    if einheit == 'µm':
        return 'mm/s', 'm/s²', 1e-3, 1e-6
    return f'{einheit}/s', f'{einheit}/s²', 1.0, 1.0

def update_sample_rate_unit():
    new_unit = "µs" if st.session_state.sample_rate_unit_toggle else "Hz"
    old_unit = st.session_state.sample_rate_unit
    if new_unit != old_unit:
        if st.session_state.sample_rate > 0:
            st.session_state.sample_rate = 1_000_000.0 / st.session_state.sample_rate
        st.session_state.sample_rate_unit = new_unit


# Alle Keys der Unter-Expander im Einstellungen-Block (Reihenfolge = Anzeigereihenfolge)
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


# Kanal-Presets je Dateityp – wird von update_sample_rate_for_file_type() genutzt
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
# HILFSFUNKTIONEN – INDEX UND GESCHWINDIGKEIT
# ---------------------------------------------------------------------------

def get_idx_at_x(x: float, sample_rate: float, max_idx: int, hz_faktor: float = 1000.0) -> int:
    """Wandelt Zeitwert (in Anzeigeeinheit) in DataFrame-Index um. O(1)."""
    return int(np.clip(round(x / hz_faktor * sample_rate), 0, max_idx))


def _berechne_ableitungen_fuer_diagramm(
    df_quelle: pd.DataFrame,
    sensor: str,
    show_velocity: bool,
    show_acceleration: bool,
    v_faktor: float = 1e-3,
    a_faktor: float = 1e-6,
    hz_faktor: float = 1000.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Berechnet D (1. Ableitung) und D2 (2. Ableitung) für die Diagramm-Darstellung.

    v_faktor/a_faktor wandeln die SG-Rohableitung in die Anzeigeeinheit um.
    Gibt (d, d2) zurück, jeweils None wenn nicht aktiviert oder zu wenig Daten.
    """
    if len(df_quelle) <= 1:
        return None, None

    arr  = df_quelle[sensor].values
    dt_s = (df_quelle['Zeit (ms)'].iloc[1] - df_quelle['Zeit (ms)'].iloc[0]) / hz_faktor

    velocity = None
    if show_velocity:
        roh = reader.berechne_sg_ableitung(arr, dt_s, st.session_state.window_length, 1)
        velocity = roh * v_faktor if roh is not None else None

    acceleration = None
    if show_acceleration:
        roh = reader.berechne_sg_ableitung(arr, dt_s, st.session_state.window_length_accel, 2)
        acceleration = roh * a_faktor if roh is not None else None

    return velocity, acceleration


def _zeichne_rechteck_fit(
    fig: go.Figure,
    rect_fit: dict,
    bereich_min: float,
    bereich_max: float,
    mit_fuellung: bool,
    yaxis: str = 'y',
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
            yaxis=yaxis,
        ))
        if mit_fuellung:
            for x_kante in (clipped_start, clipped_end):
                fig.add_shape(
                    type='line',
                    x0=x_kante, x1=x_kante,
                    y0=rect_fit['y_low'], y1=rect_fit['y_high'],
                    line=dict(color=FARBE_RECHTECK, width=1, dash='dash'),
                    yref=yaxis,
                )
            fig.add_shape(
                type='rect',
                x0=clipped_start, x1=clipped_end,
                y0=rect_fit['y_low'], y1=rect_fit['y_high'],
                line=dict(width=0),
                fillcolor='rgba(0,255,0,0.08)',
                yref=yaxis,
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
    v_faktor: float = 1e-3,
) -> tuple[list, float]:
    """Findet SOP-Punkte an steigenden Flanken des Rechteck-Fits.

    Gibt (sop_linien, v_sop) zurück:
    - sop_linien: Liste von (t_sop, t_links, t_rechts, y_level) für Diagramm-Linien
    - v_sop:      D am ersten Kreuzungspunkt (in Anzeigeeinheit), oder nan
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
        v_sop = ((signal[i1] - signal[i0]) * v_faktor) / dt_s if dt_s > 0 else float('nan')

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
    kanal_einheit_map: dict | None = None,
    alle_sensor_namen: list[str] | None = None,
    hz_faktor: float = 1000.0,
    zeit_einheit: str = 'ms',
) -> bytes:
    """Rendert das Diagramm mit Kaleido zu PNG-Bytes für den Export."""
    if kanal_einheit_map is None:
        kanal_einheit_map = {n: 'µm' for n in sensor_namen}

    _aktiv_e_e = kanal_einheit_map.get(active_sensor, 'µm')
    v_einheit_e, a_einheit_e, v_faktor_e, a_faktor_e = _ableit_info(_aktiv_e_e)

    # Y-Achse: 15 % Puffer – nur Kanäle der primären Einheit berücksichtigen
    _prim_e = next(iter(dict.fromkeys(kanal_einheit_map.get(n, 'µm') for n in sensor_namen)), 'µm')
    _prim_n = [n for n in sensor_namen if kanal_einheit_map.get(n, 'µm') == _prim_e] or sensor_namen
    y_max_e   = float(df[_prim_n].max().max())
    y_min_e   = float(df[_prim_n].min().min())
    y_range_e = [y_min_e, y_max_e + (y_max_e - y_min_e) * 0.15]

    # Ableitungen für Export-Diagramm berechnen
    velocity = acceleration = None
    if len(df) > 1:
        arr  = df[active_sensor].values
        dt_s = (df['Zeit (ms)'].iloc[1] - df['Zeit (ms)'].iloc[0]) / hz_faktor
        if show_velocity:
            roh = reader.berechne_sg_ableitung(arr, dt_s, window_length, 1)
            velocity = roh * v_faktor_e if roh is not None else None
        if show_acceleration:
            roh = reader.berechne_sg_ableitung(arr, dt_s, window_length_accel, 2)
            acceleration = roh * a_faktor_e if roh is not None else None

    velocity_ok_e     = velocity is not None
    acceleration_ok_e = acceleration is not None
    _alle_e = alle_sensor_namen if alle_sensor_namen is not None else sensor_namen
    _kanal_farbe_e = {name: KANAL_FARBEN[_alle_e.index(name) if name in _alle_e else 0]
                     for name in sensor_namen}
    _kanal_bereiche_e: dict[str, tuple[float, float]] = {
        n: (float(df[n].min()), float(df[n].max())) for n in sensor_namen if n in df.columns
    }
    kanal_zu_yaxis_e, layout_yachsen_e, v_yaxis_e, a_yaxis_e, x_domain_end_e = _yachsen_layout(
        sensor_namen, kanal_einheit_map, y_range_e,
        show_velocity, velocity_ok_e, show_acceleration, acceleration_ok_e,
        v_einheit=v_einheit_e, a_einheit=a_einheit_e,
        kanal_farbe_map=_kanal_farbe_e,
        kanal_bereiche=_kanal_bereiche_e,
    )
    active_yaxis_e = kanal_zu_yaxis_e.get(active_sensor, 'y')

    export_fig = go.Figure()

    _alle = alle_sensor_namen if alle_sensor_namen is not None else sensor_namen
    for name in sensor_namen:
        _ci = _alle.index(name) if name in _alle else 0
        export_fig.add_trace(go.Scatter(
            x=df['Zeit (ms)'], y=df[name],
            name=name, line=dict(color=KANAL_FARBEN[_ci]),
            yaxis=kanal_zu_yaxis_e.get(name, 'y'),
        ))

    export_fig.add_vline(x=xa, line_dash="dash", line_color=FARBE_CURSOR)
    export_fig.add_vline(x=xb, line_dash="dash", line_color=FARBE_CURSOR)

    if show_v_avg:
        export_fig.add_trace(go.Scatter(
            x=[xa, xb], y=[ya, yb],
            mode='lines+markers', name='Schnittlinie',
            line=dict(color=FARBE_V_SCHNITT, width=2, dash='dot'),
            yaxis=active_yaxis_e,
        ))

    if rect_fit is not None:
        _zeichne_rechteck_fit(
            export_fig, rect_fit,
            df['Zeit (ms)'].min(), df['Zeit (ms)'].max(),
            mit_fuellung=show_rect_fit, yaxis=active_yaxis_e,
        )

    if has_vmax and t_vmax_start is not None:
        export_fig.add_trace(go.Scatter(
            x=[t_vmax_start, t_vmax_ende], y=[y_vmax_start, y_vmax_ende],
            mode='lines+markers', name='D-max',
            line=dict(color=FARBE_VMAX, width=2),
            yaxis=active_yaxis_e,
        ))
    if has_amax_falling and t_amax_falling is not None:
        export_fig.add_trace(go.Scatter(
            x=[t_amax_falling], y=[y_amax_falling],
            mode='markers', name='D2-max',
            marker=dict(color=FARBE_AMAX, size=14, symbol='cross',
                        line=dict(color=FARBE_AMAX, width=2)),
            yaxis=active_yaxis_e,
        ))
    if has_amax_rising and t_amax_rising is not None:
        export_fig.add_trace(go.Scatter(
            x=[t_amax_rising], y=[y_amax_rising],
            mode='markers', name='D2-min',
            marker=dict(color=FARBE_AMAX, size=12, symbol='circle',
                        line=dict(color=FARBE_AMAX, width=2)),
            yaxis=active_yaxis_e,
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
                line=dict(color=FARBE_D, width=2),
                yaxis=active_yaxis_e,
            ))
            export_fig.add_trace(go.Scatter(
                x=[t_sop], y=[y_lvl],
                mode='markers', showlegend=False,
                marker=dict(color=FARBE_D, size=14, symbol='x',
                            line=dict(color=FARBE_D, width=2)),
                yaxis=active_yaxis_e,
            ))
            erste_sichtbar = False
    if show_velocity and velocity is not None:
        export_fig.add_trace(go.Scatter(
            x=df['Zeit (ms)'], y=velocity,
            name='D', yaxis=v_yaxis_e, line=dict(color=FARBE_D),
        ))
    if show_acceleration and acceleration is not None:
        export_fig.add_trace(go.Scatter(
            x=df['Zeit (ms)'], y=acceleration,
            name='D2', yaxis=a_yaxis_e, line=dict(color=FARBE_D2),
        ))

    export_fig.update_layout(
        xaxis_title=f"Zeit ({zeit_einheit})",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
        xaxis=dict(autorange=True, rangemode='nonnegative', domain=[0, x_domain_end_e]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        **layout_yachsen_e,
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
# Zeiteinheit früh aus Session-State lesen – wird in der gesamten Sidebar benötigt
_zhf          = st.session_state.get('zeit_hz_faktor', 1000.0)
_ZEIT_EINHEIT_MAP = {1.0: 's', 1e3: 'ms', 1e6: 'µs', 1e9: 'ns'}
_zeit_einheit = _ZEIT_EINHEIT_MAP.get(_zhf, 'ms')

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
            # Platzhalter-Wert damit sample_rate weiter unten verfügbar ist
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

    # Kanalanzahl aus Datei erkennen (gecacht) – steuert wie viele Felder der Expander zeigt
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

    # Vor Widget-Rendering: leere Kanalnamen für erkannte Datei-Kanäle auffüllen
    if uploaded_file and _n_show > 0:
        for _i in range(1, _n_show + 1):
            if not st.session_state.get(f'ch{_i}_name', '').strip():
                st.session_state[f'ch{_i}_name'] = f'Kanal {_i}'

    with st.expander("Kanäle", expanded=st.session_state.sub_kanaele, key="sub_kanaele", on_change=_SUB_EXPANDER_CBS['sub_kanaele']):
        st.caption("Leer: wird beim Schließen automatisch als 'Kanal N' benannt.")
        # Für Oszilloskop: Einheiten aus Datei-Header vorlesen
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
        kanal_namen_tuple = tuple(n for n in _kanal_cfg if n)
        # Kanalname → 1-basierte Kanal-Nummer (für show_chN Keys)
        _sensor_ch_num = {name: i + 1 for i, name in enumerate(_kanal_cfg) if name}

        if len(kanal_namen_tuple) < 1:
            st.sidebar.error("Mindestens ein Kanalname muss angegeben werden.")
            st.stop()

        sensor_namen = list(kanal_namen_tuple)
        file_bytes = uploaded_file.getvalue()
        _osc_skale_tuple = tuple(
            st.session_state[f'osc_skale_{i+1}'] for i in range(len(kanal_namen_tuple))
        )

        try:
            df_raw, _hz_f = load_rohdaten(
                file_bytes, file_type, st.session_state.skip_rows,
                st.session_state.max_samples, kanal_namen_tuple, _osc_skale_tuple,
            )
            st.session_state['zeit_hz_faktor'] = _hz_f
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
            # Überzählige Kanalnamen leeren (neue Datei hat evtl. weniger Kanäle)
            _n_d = st.session_state.get('n_kanäle_datei', N_KANÄLE)
            for _i in range(_n_d + 1, N_KANÄLE + 1):
                st.session_state[f'ch{_i}_name'] = ''
            # Alle erkannten Kanäle einblenden
            for _i in range(1, _n_show + 1):
                st.session_state[f'show_ch{_i}'] = True
            st.session_state.xa    = total_time_ms * 0.30
            st.session_state.xa_sw = total_time_ms * 0.30
            st.session_state.xa_nw = total_time_ms * 0.30
            st.session_state.xb    = total_time_ms * 0.60
            st.session_state.xb_sw = total_time_ms * 0.60
            st.session_state.xb_nw = total_time_ms * 0.60
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

    # Y-Offset und X-Offset außerhalb des if-Blocks – Expander müssen immer gerendert
    # werden damit die Akkordeon-Callbacks zuverlässig funktionieren.
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
                _off_lim  = max(600.0,
                                abs(float(_raw_col.min())) * 1.5,
                                abs(float(_raw_col.max())) * 1.5)
                _off_lim  = float(np.ceil(_off_lim / 100) * 100)
                _off_step = (0.1  if _off_lim <=   600 else
                             1.0  if _off_lim <=  6000 else
                             10.0 if _off_lim <= 60000 else 100.0)
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

    # Aktive Einheiten aus konfigurierten Kanälen bestimmen
    _grenzw_einheiten = list(dict.fromkeys(
        st.session_state.get(f'ch{_i}_einheit', 'µm')
        for _i in range(1, N_KANÄLE + 1)
        if st.session_state.get(f'ch{_i}_name', '').strip()
    )) or ['µm']

    with st.expander("Diagramm-Grenzwerte", expanded=st.session_state.sub_grenzwerte, key="sub_grenzwerte", on_change=_SUB_EXPANDER_CBS['sub_grenzwerte']):
        for _e in _grenzw_einheiten:
            st.caption(_e)
            _gc1, _gc2 = st.columns(2)
            _gc1.number_input(f"min ({_e})", step=1.0, format="%.2f",
                              key=einheit_ss_key_min(_e), label_visibility="collapsed")
            _gc2.number_input(f"max ({_e})", step=1.0, format="%.2f",
                              key=einheit_ss_key_max(_e), label_visibility="collapsed")
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
# _kanal_cfg und kanal_namen_tuple wurden bereits im Sidebar-Block gesetzt.
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
# DATEN LADEN
# ---------------------------------------------------------------------------

file_bytes = uploaded_file.getvalue()
osc_skale_tuple = tuple(
    st.session_state[f'osc_skale_{i+1}'] for i in range(len(kanal_namen_tuple))
)

try:
    df_raw, _hz_faktor = load_rohdaten(
        file_bytes, file_type, st.session_state.skip_rows,
        st.session_state.max_samples, kanal_namen_tuple, osc_skale_tuple,
    )
    st.session_state['zeit_hz_faktor'] = _hz_faktor
except ValueError as e:
    st.error(f"Fehler beim Laden: {e}")
    st.stop()

sensor_namen = list(kanal_namen_tuple)   # tatsächlich geladene Kanalnamen

# ---------------------------------------------------------------------------
# DATENAUFBEREITUNG – reader.build_display_df erzeugt df_full + aktuelle Samplerate
# ---------------------------------------------------------------------------

# Offsets für alle aktiven Kanäle auslesen
offs = tuple(st.session_state[f'off{i+1}'] for i in range(len(sensor_namen)))

_zhf = st.session_state.get('zeit_hz_faktor', 1000.0)   # s → Anzeigeeinheit
_ZEIT_EINHEIT_MAP = {1.0: 's', 1e3: 'ms', 1e6: 'µs', 1e9: 'ns'}
_zeit_einheit = _ZEIT_EINHEIT_MAP.get(_zhf, 'ms')

df_full, sample_rate = reader.build_display_df(
    df_raw, file_type, sample_rate, kanal_namen_tuple, offs,
    zeit_hz_faktor=_zhf,
)

# ---------------------------------------------------------------------------
# USE-DATA – X-Offset als Sample-Shift einbauen
# RAW → [scale + Y-offset] = df_full   →   [X-shift] = df_use   →   [Crop] = df
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
    st.session_state.xa    = total_time_ms * 0.30
    st.session_state.xa_sw = total_time_ms * 0.30
    st.session_state.xa_nw = total_time_ms * 0.30
    st.session_state.xb    = total_time_ms * 0.60
    st.session_state.xb_sw = total_time_ms * 0.60
    st.session_state.xb_nw = total_time_ms * 0.60
    st.session_state.crop_start     = None
    st.session_state.crop_end       = None
    st.session_state.zoom_token    += 1
    st.session_state.last_file_name = uploaded_file.name
    st.rerun()

max_zeit_full = float(df_use['Zeit (ms)'].iloc[-1])
max_idx_full  = len(df_use) - 1

# ---------------------------------------------------------------------------
# CROP-LOGIK
# ---------------------------------------------------------------------------

crop_active = (
    st.session_state.crop_start is not None
    and st.session_state.crop_end is not None
)
if crop_active:
    ci_start = get_idx_at_x(st.session_state.crop_start, sample_rate, max_idx_full, _zhf)
    ci_end   = get_idx_at_x(st.session_state.crop_end,   sample_rate, max_idx_full, _zhf)
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

# Ableitungs-Einheiten und Konversionsfaktoren für den aktiven Kanal
_aktiv_einheit = kanal_einheit_map.get(active_sensor, 'µm')
v_einheit, a_einheit, v_faktor, a_faktor = _ableit_info(_aktiv_einheit)

# Cursor-Werte auf aktiven Zeitbereich begrenzen.
# Alle drei gebundenen Keys (xa, xa_sw, xa_nw) werden synchronisiert damit
# Slider und Zahlenfeld nie voneinander abweichen.
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
    _tb_f = _zhf / 1e3   # ms → aktuelle Zeiteinheit (nur Rückrechnung)
    _v_tb_display = st.slider(
        f"Zeitbasis D-max ({_zeit_einheit})",
        0.010, 0.100, 0.030,
        step=0.005, format=f"%.3f {_zeit_einheit}",
        help="Mittelungsfenster für D-max, D2-max und SOP: Der Peak wird über dieses Zeitfenster gemittelt. Kleiner = empfindlicher, größer = robuster gegenüber Rauschen.",
    )
    v_time_base_ms = _v_tb_display / _tb_f

show_v_avg    = st.sidebar.toggle("Schnittlinie A–B anzeigen", key="show_v_avg",
                                  help="Zeichnet eine Verbindungslinie von XA nach XB und visualisiert damit die mittlere Änderungsrate D (A-B).")
show_rect_fit = st.sidebar.toggle(
    "Rechteck-Fit füllen", key="show_rect_fit",
    help="Zeigt zusätzlich vertikale Kantenlinien und hellgrüne Füllung für alle erkannten Rechteck-Pulse.",
)
show_velocity = st.sidebar.toggle(
    "D anzeigen (1. Ableitung)", key="show_velocity",
    help="Zeigt die 1. Ableitung des aktiven Kanals auf einer zweiten Y-Achse rechts.",
)
if show_velocity:
    st.sidebar.slider(
        "Glättung D", 5, 80, step=1,
        value=st.session_state.window_length,
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
        value=st.session_state.window_length_accel,
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
        value=st.session_state.sop_percent,
        key="sop_percent",
        help="Höhe auf der steigenden Flanke in Prozent des Hub (0 % = unterer Pegel, 100 % = oberer Pegel).",
    )

# Rechteck-Fit auf den vollständigen (ungecropten) Datensatz anwenden
rect_fit = compute_best_fit_rectangle(
    df_use['Zeit (ms)'].values,
    df_use[active_sensor].values,
)

# ---------------------------------------------------------------------------
# MESSWERTBERECHNUNG
# ---------------------------------------------------------------------------

# Indizes der Cursor-Positionen im (ggf. gecropten) DataFrame
if crop_active:
    idx_a = get_idx_at_x(xa - min_zeit, sample_rate, max_idx, _zhf)
    idx_b = get_idx_at_x(xb - min_zeit, sample_rate, max_idx, _zhf)
else:
    idx_a = get_idx_at_x(xa, sample_rate, max_idx, _zhf)
    idx_b = get_idx_at_x(xb, sample_rate, max_idx, _zhf)

ya = df.loc[idx_a, active_sensor]
yb = df.loc[idx_b, active_sensor]

dt_val_ms = abs(xb - xa)                                                    # Anzeigeeinheit
dy        = abs(yb - ya)                                                    # [aktiv_einheit]
# v_avg: dy/dt in [einheit/Anzeigeeinheit] → × hz_faktor → Anzeigeeinheit/s → × v_faktor
v_avg     = dy / dt_val_ms * _zhf * v_faktor if dt_val_ms > 0 else 0.0

# Momentan-D an XA und XB über ein Zeitbasis-Fenster
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
    gefilt_geschw_roh_full = reader.berechne_sg_ableitung(arr_full, dt_step_s, st.session_state.window_length, 1)
    if gefilt_geschw_roh_full is not None:
        abs_geschw_full = np.abs(gefilt_geschw_roh_full * v_faktor)
        # Peak nur im Cursor-Bereich suchen
        abs_geschw_slice  = abs_geschw_full[idx_start:idx_end + 1]
        idx_vmax_peak_loc = int(np.argmax(abs_geschw_slice))
        idx_vmax_peak     = idx_start + idx_vmax_peak_loc
        iv_start          = max(0, idx_vmax_peak - halbes_zeitfenster)
        iv_ende           = min(max_idx, idx_vmax_peak + halbes_zeitfenster)
        v_max             = float(np.mean(abs_geschw_full[iv_start:iv_ende + 1]))

        if 0 <= iv_start <= max_idx and 0 <= iv_ende <= max_idx:
            t_vmax_start = df.loc[iv_start, 'Zeit (ms)']
            y_vmax_start = df.loc[iv_start, active_sensor]
            t_vmax_ende  = df.loc[iv_ende,  'Zeit (ms)']
            y_vmax_ende  = df.loc[iv_ende,  active_sensor]
            has_vmax     = True

    # a-max: SG-Filter auf dem vollständigen Datensatz – verhindert Randeffekte an Cursor-Grenzen
    gefilt_beschl_roh_full = reader.berechne_sg_ableitung(arr_full, dt_step_s, st.session_state.window_length_accel, 2)
    if gefilt_beschl_roh_full is not None:
        gefilt_beschl_full = gefilt_beschl_roh_full * a_faktor

        def _peak_marker(idx_abs):
            """Gemittelter Beschleunigungswert und Diagramm-Position für einen Peak (absoluter Index)."""
            ia0  = max(0, idx_abs - halbes_zeitfenster)
            ia1  = min(max_idx, idx_abs + halbes_zeitfenster)
            wert = float(np.mean(gefilt_beschl_full[ia0:ia1 + 1]))
            return wert, float(df.loc[idx_abs, 'Zeit (ms)']), float(df.loc[idx_abs, active_sensor])

        # Peak nur im Cursor-Bereich suchen
        beschl_slice = gefilt_beschl_full[idx_start:idx_end + 1]

        idx_falling_abs                                = idx_start + int(np.argmax(beschl_slice))
        a_max_falling, t_amax_falling, y_amax_falling = _peak_marker(idx_falling_abs)
        has_amax_falling = True

        idx_rising_abs                               = idx_start + int(np.argmin(beschl_slice))
        a_min_rising, t_amax_rising, y_amax_rising   = _peak_marker(idx_rising_abs)
        has_amax_rising = True

# SOP – steht nach halbes_zeitfenster-Definition und nach rect_fit
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
    step    = len(df) // MAX_PLOT_PUNKTE
    df_plot = df.iloc[::step]
else:
    df_plot = df

# ---------------------------------------------------------------------------
# ABLEITUNGEN FÜR DIAGRAMM-DARSTELLUNG
# ---------------------------------------------------------------------------

velocity, acceleration = _berechne_ableitungen_fuer_diagramm(
    df_plot, active_sensor, show_velocity, show_acceleration,
    v_faktor=v_faktor, a_faktor=a_faktor, hz_faktor=_zhf,
)

# ---------------------------------------------------------------------------
# DIAGRAMM AUFBAUEN
# ---------------------------------------------------------------------------

# Y-Bereiche aus vollständigen Daten (df_use) berechnen – Crop darf die Y-Achse nicht verschieben
_prim_einheit = next(iter(dict.fromkeys(kanal_einheit_map.get(n, 'µm') for n in sichtbare_sensor_namen)), 'µm')
_prim_namen   = [n for n in sichtbare_sensor_namen if kanal_einheit_map.get(n, 'µm') == _prim_einheit] or sichtbare_sensor_namen

# Primäre Achse: Bereich aus vollständigem Datensatz
_prim_namen_full = [n for n in _prim_namen if n in df_use.columns]
_y_full_prim = df_use[_prim_namen_full] if _prim_namen_full else None
if _y_full_prim is not None and not _y_full_prim.empty:
    y_max_plot = float(_y_full_prim.max().max())
    y_min_plot = float(_y_full_prim.min().min())
else:
    y_max_plot = float(df_plot[_prim_namen].max().max())
    y_min_plot = float(df_plot[_prim_namen].min().min())
y_range_plot = [y_min_plot, y_max_plot + (y_max_plot - y_min_plot) * 0.15]

# Fallback-Bereiche je Einheit aus vollständigem Datensatz – für Sekundärachsen
_yrange_fallback: dict[str, list] = {}
for _e in set(kanal_einheit_map.get(n, 'µm') for n in sichtbare_sensor_namen):
    _cols_e = [n for n in sichtbare_sensor_namen if kanal_einheit_map.get(n, 'µm') == _e and n in df_use.columns]
    if _cols_e:
        _lo_e = float(df_use[_cols_e].min().min())
        _hi_e = float(df_use[_cols_e].max().max())
        _span_e = _hi_e - _lo_e
        _yrange_fallback[_e] = [_lo_e, _hi_e + _span_e * 0.15]

# Per-Kanal-Bereiche für automatische Achsen-Aufteilung (SPLIT_FAKTOR)
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
)
active_yaxis = kanal_zu_yaxis.get(active_sensor, 'y')

fig = go.Figure()

for name in sichtbare_sensor_namen:
    _ci = sensor_namen.index(name)
    fig.add_trace(go.Scatter(
        x=df_plot['Zeit (ms)'], y=df_plot[name],
        name=name, line=dict(color=KANAL_FARBEN[_ci]),
        yaxis=kanal_zu_yaxis.get(name, 'y'),
    ))

fig.add_vline(x=xa, line_dash="dash", line_color=FARBE_CURSOR)
fig.add_vline(x=xb, line_dash="dash", line_color=FARBE_CURSOR)

if show_v_avg:
    fig.add_trace(go.Scatter(
        x=[xa, xb], y=[ya, yb],
        mode='lines+markers', name='Schnittlinie',
        line=dict(color=FARBE_V_SCHNITT, width=2, dash='dot'),
        yaxis=active_yaxis,
    ))

if rect_fit is not None:
    _zeichne_rechteck_fit(fig, rect_fit, min_zeit, max_zeit,
                          mit_fuellung=show_rect_fit, yaxis=active_yaxis)

if has_vmax:
    fig.add_trace(go.Scatter(
        x=[t_vmax_start, t_vmax_ende], y=[y_vmax_start, y_vmax_ende],
        mode='lines+markers', name='D-max',
        line=dict(color=FARBE_VMAX, width=2),
        yaxis=active_yaxis,
    ))
if has_amax_falling:
    fig.add_trace(go.Scatter(
        x=[t_amax_falling], y=[y_amax_falling],
        mode='markers', name='D2-max',
        marker=dict(color=FARBE_AMAX, size=14, symbol='cross',
                    line=dict(color=FARBE_AMAX, width=2)),
        yaxis=active_yaxis,
    ))
if has_amax_rising:
    fig.add_trace(go.Scatter(
        x=[t_amax_rising], y=[y_amax_rising],
        mode='markers', name='D2-min',
        marker=dict(color=FARBE_AMAX, size=12, symbol='circle',
                    line=dict(color=FARBE_AMAX, width=2)),
        yaxis=active_yaxis,
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
            line=dict(color=FARBE_D, width=2),
            yaxis=active_yaxis,
        ))
        fig.add_trace(go.Scatter(
            x=[t_sop], y=[y_lvl],
            mode='markers', showlegend=False,
            marker=dict(color=FARBE_D, size=14, symbol='x',
                        line=dict(color=FARBE_D, width=2)),
            yaxis=active_yaxis,
        ))
        erste_sichtbar = False
if show_velocity and velocity is not None:
    fig.add_trace(go.Scatter(
        x=df_plot['Zeit (ms)'], y=velocity,
        name='D', yaxis=v_yaxis, line=dict(color=FARBE_D),
    ))
if show_acceleration and acceleration is not None:
    fig.add_trace(go.Scatter(
        x=df_plot['Zeit (ms)'], y=acceleration,
        name='D2', yaxis=a_yaxis, line=dict(color=FARBE_D2),
    ))

_y_lim_keys = (
    ['v_axis_min', 'v_axis_max', 'a_axis_min', 'a_axis_max']
    + [einheit_ss_key_min(e) for e in EINHEIT_ALLE]
    + [einheit_ss_key_max(e) for e in EINHEIT_ALLE]
)
_y_lim_token  = hash(tuple(st.session_state.get(k, 0) for k in _y_lim_keys))
# Achsen-Struktur-Token: erzwingt Plotly-Reset wenn sich Kanal↔Achse-Zuordnung ändert
_axis_token   = hash(tuple(sorted(kanal_zu_yaxis.items())))

fig.update_layout(
    xaxis_title=f"Zeit ({_zeit_einheit})",
    height=600,
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
    uirevision=f"{st.session_state.zoom_token}-{st.session_state.crop_start}-{st.session_state.crop_end}-{_y_lim_token}-{_axis_token}",
    xaxis=dict(autorange=True, rangemode='nonnegative', domain=[0, x_domain_end]),
    **layout_yachsen,
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
        key="xa_sw", step=0.001, format=f"%.3f {_zeit_einheit}",
        on_change=update_xa_from_slider, label_visibility="collapsed",
        help=f"Linker Cursor XA ({_zeit_einheit}) – ziehen oder Wert im Expander 'Zeitmarker & Basis' eingeben.",
    )
    st.slider(
        "XB", min_zeit, max_zeit, value=xb,
        key="xb_sw", step=0.001, format=f"%.3f {_zeit_einheit}",
        on_change=update_xb_from_slider, label_visibility="collapsed",
        help=f"Rechter Cursor XB ({_zeit_einheit}) – ziehen oder Wert im Expander 'Zeitmarker & Basis' eingeben.",
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

freq_hz = (_zhf / dt_val_ms) if dt_val_ms > 0 else float('nan')
hub     = abs(rect_fit['y_high'] - rect_fit['y_low']) if rect_fit is not None else float('nan')

# Zeile 1 – Zeit & Signal
z1, z2, z3, z4 = st.columns(4)
z1.metric("Δt (A-B)",          f"{dt_val_ms:.3f} ms")
z2.metric("Frequenz Δt (A-B)", f"{freq_hz:.1f} Hz"          if not np.isnan(freq_hz) else "N/A")
z3.metric("Δs (A-B)",          f"{dy:.1f} {_aktiv_einheit}")
z4.metric("Hub Best-fit",      f"{hub:.1f} {_aktiv_einheit}" if not np.isnan(hub) else "N/A")

# Zeile 2 – D (1. Ableitung)
g1, g2, g3, g4 = st.columns(4)
g1.metric("D (A-B)",           f"{v_avg:.1f} {v_einheit}")
g2.metric("ΔD (A-B)",          f"{v_cursor_delta:.1f} {v_einheit}" if not np.isnan(v_cursor_delta) else "N/A")
g3.metric("D max (Peak)",      f"{v_max:.1f} {v_einheit}"          if not np.isnan(v_max) else "N/A")
g4.metric("SOP",               f"{v_sop:.1f} {v_einheit}"          if not np.isnan(v_sop) else "N/A")

# Zeile 3 – D2 (2. Ableitung)
a1, a2 = st.columns(2)
a1.metric("D2 max Fall.",      f"{a_max_falling:.1f} {a_einheit}"  if not np.isnan(a_max_falling) else "N/A")
a2.metric("D2 min Rise.",      f"{a_min_rising:.1f} {a_einheit}"   if not np.isnan(a_min_rising) else "N/A")

# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------

st.sidebar.header("3. Export")
metrics = {
    # Zeit & Signal
    "XA (ms)":              f"{xa:.3f}",
    "XB (ms)":              f"{xb:.3f}",
    "Δt (A-B)":             f"{dt_val_ms:.3f} ms",
    "Frequenz Δt (A-B)":    f"{freq_hz:.1f} Hz"                        if not np.isnan(freq_hz) else "N/A",
    "Δs (A-B)":             f"{dy:.1f} {_aktiv_einheit}",
    "Hub Best-fit":         f"{hub:.1f} {_aktiv_einheit}"              if not np.isnan(hub) else "N/A",
    # D (1. Ableitung)
    "D (A-B)":              f"{v_avg:.1f} {v_einheit}",
    f"ΔD (A-B)":            f"{v_cursor_delta:.1f} {v_einheit}"        if not np.isnan(v_cursor_delta) else "N/A",
    "D max (Peak)":         f"{v_max:.1f} {v_einheit}"                 if not np.isnan(v_max) else "N/A",
    "SOP":                  f"{v_sop:.1f} {v_einheit}"                 if not np.isnan(v_sop) else "N/A",
    # D2 (2. Ableitung)
    "D2 max Fall.":         f"{a_max_falling:.1f} {a_einheit}"         if not np.isnan(a_max_falling) else "N/A",
    "D2 min Rise.":         f"{a_min_rising:.1f} {a_einheit}"          if not np.isnan(a_min_rising) else "N/A",
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
                hz_faktor=_zhf,
                zeit_einheit=_zeit_einheit,
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
