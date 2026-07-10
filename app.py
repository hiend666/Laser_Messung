# pip install streamlit pandas plotly scipy "kaleido==0.2.1" reportlab
# streamlit run app.py
"""
Messdaten-Auswertung – Laservibrometer CSV.

Zeigt Signal-, D- und D2-Kurven (1. und 2. Ableitung) und
exportiert Ergebnisse als PDF oder PNG.

ARCHITEKTUR-REGELN (Kurzreferenz):

SI-Zeitregel: Alle Zeitwerte intern als Sekunden (float64) in Spalte 'Zeit (s)'.
Anzeigekonvertierung ausschließlich an UI-Grenzen:
  _sc(t) = (t - t_offset) × _vis_scale  →  Anzeigewert in _vis_einheit
  _vis_scale = _vis_hz (z.B. 1000 für ms, 1e6 für µs, automatisch aus crop-Bereich)

Display-Kopie-Muster: df_plot_c und rect_fit_c sind reine Anzeigeversionen.
  df_plot_c hat 'Zeit'-Spalte in Anzeigeeinheit (kein 'Zeit (s)').
  rect_fit_c hat t_start/t_end per _sc() konvertiert, weil _zeichne_rechteck_fit
  t-Werte direkt als Chart-x-Koordinaten verwendet.

Zwei-Key-Muster (Session State): freie Keys (xa, xb, off1, …) = Wahrheitsquelle,
  Widget-Keys (*_sw, *_nw) gehören ausschließlich Streamlit-Widgets.
  on_change-Callbacks schreiben von Widget-Key → freien Key; nie umgekehrt.
  Ausnahmen (Widget = Wahrheitsquelle): ya_sw, yb_sw, off_kanal_sel, active_sensor_name.

Integrale: _fmt_integral-Skalare integrieren über 'Zeit (s)' → Einheit [aktiv·s].
  Kurvenintegrale im Chart nutzen dt aus df_plot_c → Einheit [aktiv·vis_einheit].
"""
import io
import json
import math
import pathlib
import sys

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Prefer the local project modules over any globally installed copies.
_APP_ROOT = pathlib.Path(__file__).resolve().parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

import reader
import chart as chart_module

KANAL_FARBEN = chart_module.KANAL_FARBEN
FARBE_CURSOR = chart_module.FARBE_CURSOR
_ZEIT_TO_S = chart_module._ZEIT_TO_S
_ableit_info = chart_module._ableit_info
_laenge_autoscale = chart_module._laenge_autoscale
_LAENGE_EINHEITEN = chart_module._LAENGE_EINHEITEN
_yachsen_layout = chart_module._yachsen_layout
_zeichne_integral_flaeche = chart_module._zeichne_integral_flaeche
_finde_sop_kreuzungen = chart_module._finde_sop_kreuzungen
_baue_traces = chart_module._baue_traces
build_chart_png = chart_module.build_chart_png
build_pdf = chart_module.build_pdf

VERSION = "v1.06.00"

# ---------------------------------------------------------------------------
# KONSTANTEN
# ---------------------------------------------------------------------------

MAX_PLOT_PUNKTE = 5_000     # Downsampling-Schwelle für interaktives Diagramm
N_KANÄLE        = 4         # Maximale Kanalanzahl – einzige Stelle um diese zu ändern
Y_PUFFER        = 0.15      # Y-Bereich-Puffer oben (15 %) – muss mit chart.Y_PUFFER übereinstimmen

# Einheiten-Auswahl für Kanäle
EINHEIT_OPTIONEN = ['µm', 'mm', 'm', 'V', 'mV', 'A', 'mA', 'N', 'kN', 'bar', 'Pa', '°C', '%']

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
# Zwei-Key-Muster – Details im Modul-Docstring oben.
# Freie Keys (xa, xb, off1, …): Wahrheitsquelle für alle berechneten Werte.
# Widget-Keys (xa_sw, xa_nw, …): gehören ausschließlich den Streamlit-Widgets;
# ihre on_change-Callbacks schreiben in den freien Key zurück.
# Externe Setter (Buttons, Auto-Reset) schreiben nur in freie Keys.
# Ausnahmen mit Einzelschlüssel (Widget = Wahrheitsquelle): ya_sw, yb_sw,
# off_kanal_sel, active_sensor_name — diese haben keine schreibenden Callbacks.
_CH_NAMEN_DEFAULT  = ['Festo', 'DST'] + [''] * (N_KANÄLE - 2)
_OSC_SKALE_DEFAULT = [1.0, 1.0, 100.0] + [1.0] * (N_KANÄLE - 3)

# Speicherbare Einstellungen: Dict ist gleichzeitig Defaults und Schlüsselliste.
# Neue saveable Keys hier eintragen – EINSTELLUNGEN_KEYS wird automatisch daraus abgeleitet.
_EINSTELLUNGEN_DEFAULTS: dict = {
    'file_type_radio': 'CSV plain',
    'xa': 0.0,
    'xb': 0.001,
    'sample_rate': 2.55,
    'sample_rate_unit': 'µs',
    'sample_rate_unit_toggle': True,
    'skip_rows': 12,
    'csv_header_aus_zeile1': False,
    'csv_einheit_aus_zeile2': False,
    'max_samples': 8000,
    'show_v_avg': False,
    'mark_vmax':  False,
    'mark_vmin':  False,
    'mark_afall': False,
    'mark_arise': False,
    'show_rect_fit_top': False,
    'show_rect_fit': False,
    'show_y_slider': False,
    'ya_sw': 0.0,
    'yb_sw': 0.0,
    'show_velocity': False,
    'window_length': 30,
    'show_acceleration': False,
    'window_length_accel': 40,
    'v_time_base_s': None,
    'show_sop': False,
    'sop_percent': 80,
    'show_integral': False,
    'show_integral_curve': False,
    'int_axis_min': 0,
    'int_axis_max': 0,
    'show_multi_diag': False,
    'multi_diag_ymin': 0,
    'multi_diag_ymax': 0,
    'off_fein_stufe':   '1',
    'x_off_fein_stufe': '0,1',
    'off_kanal_sel': '',
    'show_multi_kanal': False,
    'multi_kanal_auswahl': [],
    'show_widerstand_integral': False,
    'widerstand_kohm': 200.0,
    'widerstand_kanal': '',
    'v_axis_min': 0,
    'v_axis_max': 0,
    'a_axis_min': 0,
    'a_axis_max': 0,
    'gleiche_nulllinie': False,
    'export_format': 'PDF',
    'export_mit_werten': False,
    # None = noch nie manuell gewählt -> beim ersten Anzeigen werden alle Werte vorausgewählt.
    # Eine Liste (auch leer []) gilt als bewusste Nutzerauswahl und wird nicht mehr automatisch
    # auf "alle" zurückgesetzt (siehe Multiselect-Logik im Export-Bereich).
    'export_wert_auswahl': None,
    'pdf_titel': '',
    'pdf_beschreibung': '',
    'crop_start': None,
    'crop_end': None,
    'x_rel_mode': False,
    'x_rel_offset': 0.0,
    'active_sensor_name': '',
}
for _i in range(1, N_KANÄLE + 1):
    _EINSTELLUNGEN_DEFAULTS[f'ch{_i}_name']    = _CH_NAMEN_DEFAULT[_i - 1]
    _EINSTELLUNGEN_DEFAULTS[f'ch{_i}_einheit'] = 'µm'
    _EINSTELLUNGEN_DEFAULTS[f'osc_skale_{_i}'] = _OSC_SKALE_DEFAULT[_i - 1]
    _EINSTELLUNGEN_DEFAULTS[f'off{_i}']        = 0.0
    _EINSTELLUNGEN_DEFAULTS[f'off{_i}_slider'] = 0.0
    _EINSTELLUNGEN_DEFAULTS[f'x_off{_i}']     = 0.0
    _EINSTELLUNGEN_DEFAULTS[f'show_ch{_i}']   = True
    _EINSTELLUNGEN_DEFAULTS[f'ch{_i}_sg_en']  = False
    _EINSTELLUNGEN_DEFAULTS[f'ch{_i}_sg_win'] = 5
    _EINSTELLUNGEN_DEFAULTS[f'ch{_i}_ymin']   = 0.0
    _EINSTELLUNGEN_DEFAULTS[f'ch{_i}_ymax']   = 0.0
for _e in EINHEIT_OPTIONEN:
    _EINSTELLUNGEN_DEFAULTS[einheit_ss_key_min(_e)] = 0.0
    _EINSTELLUNGEN_DEFAULTS[einheit_ss_key_max(_e)] = 0.0

EINSTELLUNGEN_KEYS: list[str] = list(_EINSTELLUNGEN_DEFAULTS.keys())

# Runtime-State: nicht gespeichert, nur für laufende Session
_RUNTIME_DEFAULTS: dict = {
    'xa_sw': 0.0,
    'xb_sw': 0.001,
    'zoom_token': 0,
    'last_file_name': None,
    'export_wert_bekannte_keys': None,
    'window_length_sw': 30,
    'window_length_accel_sw': 40,
    'v_time_base_s_sw': None,
    'zeit_hz_faktor': 1000.0,
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

for _k, _v in {**_EINSTELLUNGEN_DEFAULTS, **_RUNTIME_DEFAULTS}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


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
    try:
        return reader.berechne_rechteck_fit(zeit, signal)
    except Exception as exc:
        st.sidebar.warning(f"Rechteck-Fit fehlgeschlagen: {exc}")
        return None


@st.cache_data
def _build_display_df_cached(
    df_raw: pd.DataFrame,
    file_type: str,
    sample_rate_hz: float,
    kanal_namen: tuple[str, ...],
    offsets: tuple[float, ...],
) -> tuple[pd.DataFrame, float]:
    """Gecachter Wrapper um reader.build_display_df – Y-Offsets und Zeitachse in Sekunden."""
    return reader.build_display_df(df_raw, file_type, sample_rate_hz, kanal_namen, offsets)


@st.cache_data
def _sg_ableitung_cached(
    arr: np.ndarray,
    dt_step_s: float,
    window_length: int,
    ordnung: int,
) -> np.ndarray | None:
    """Gecachter Wrapper um reader.berechne_sg_ableitung – kein Recompute bei Cursor-Bewegung."""
    return reader.berechne_sg_ableitung(arr, dt_step_s, window_length, ordnung)


@st.cache_data
def _apply_sg_vorfilter_cached(
    df_raw: pd.DataFrame,
    kanal_namen: tuple[str, ...],
    sg_params: tuple[tuple[bool, int], ...],
) -> pd.DataFrame:
    """Filtert Rohkanäle vor Y-Offset-Anwendung. Cache-Miss nur bei Daten- oder Parameteränderung."""
    if not any(en for en, _ in sg_params):
        return df_raw.copy()
    df = df_raw.copy()
    for name, (en, win) in zip(kanal_namen, sg_params):
        if en and name in df.columns:
            df[name] = reader.glaette_signal(df[name].values, win)
    return df


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
        t_off  = float(st.session_state.get('_t_offset', 0.0))
        _scale = float(st.session_state.get('_vis_scale', 1.0))
        val_rel = float(st.session_state[from_key])
        # val_rel in Anzeigeeinheit; / _vis_hz (= _scale) → Sekunden, + t_off_s → absolut
        st.session_state[cursor] = round(val_rel / _scale + t_off, 9)
        st.session_state[to_key] = val_rel
    return _cb

update_xa_from_slider = _make_cursor_cb('xa', 'slider')
update_xa_from_num    = _make_cursor_cb('xa', 'num')
update_xb_from_slider = _make_cursor_cb('xb', 'slider')
update_xb_from_num    = _make_cursor_cb('xb', 'num')

def _make_off_cb(i: int):
    def _cb(): st.session_state[f'off{i}'] = st.session_state[f'off{i}_slider']
    return _cb

OFF_CALLBACKS = [_make_off_cb(i) for i in range(1, N_KANÄLE + 1)]

def _on_window_length_cb():
    st.session_state['window_length'] = st.session_state['window_length_sw']

def _on_window_length_accel_cb():
    st.session_state['window_length_accel'] = st.session_state['window_length_accel_sw']


def _on_v_time_base_cb():
    # Widget-Wert (v_time_base_s_sw) liegt in Slider-Anzeigeeinheit vor (z.B. ms).
    # _tb_hz rechnet zurück in Sekunden (Sekunden × _tb_hz = Anzeigewert).
    # float64 mit voller Auflösung (~15 Stellen) – mind. 9 Nachkommastellen
    # für µs-Bereich bei kleinen Crops (z.B. 0.000005 s = 5 µs).
    _hz = float(st.session_state.get('_tb_hz', 1.0))
    st.session_state['v_time_base_s'] = float(np.float64(st.session_state['v_time_base_s_sw']) / np.float64(_hz))

def _on_x_rel_mode_change():
    """Friert den Nullpunkt ein wenn Start@0 aktiviert wird; löscht ihn beim Deaktivieren."""
    if st.session_state.get('x_rel_mode', False):
        _xa = float(st.session_state.get('xa', 0.0))
        _xb = float(st.session_state.get('xb', 0.001))
        st.session_state['x_rel_offset'] = min(_xa, _xb)
    else:
        st.session_state['x_rel_offset'] = 0.0

def _on_fein_toggle():
    """Beim Stufenwechsel: freie Off-Keys in Widget-Keys übernehmen,
    damit Streamlit den Slider-Wert nicht auf min_value snappt."""
    for _i in range(1, N_KANÄLE + 1):
        if f'off{_i}' in st.session_state:
            st.session_state[f'off{_i}_slider'] = st.session_state[f'off{_i}']

_FEIN_STUFEN  = ['100', '10', '1', '0,1', '0,01']
_FEIN_SCHRITT = {'100': 100.0, '10': 10.0, '1': 1.0, '0,1': 0.1, '0,01': 0.01}


def _setze_cursor_position(total_time_s: float) -> None:
    """Setzt XA und XB auf Standardposition (30 % und 60 % der Gesamtzeit in Sekunden)."""
    xa_init = round(total_time_s * CURSOR_XA_INIT_FRAC, 9)
    xb_init = round(total_time_s * CURSOR_XB_INIT_FRAC, 9)
    st.session_state.xa    = xa_init
    st.session_state.xa_sw = xa_init
    st.session_state.xb    = xb_init
    st.session_state.xb_sw = xb_init


def _fmt_zeit(value: float, einheit: str) -> str:
    """Formatiert Zeitwert mit automatisch gewählter Einheit (s/ms/µs/ns).
    Wechselt auf die nächst-kleinere Einheit sobald der Wert unter 0,9 liegt.
    """
    val_s = value * _ZEIT_TO_S.get(einheit, 1e-3)
    if val_s == 0:
        return f"0.000 {einheit}"
    for unit in ('s', 'ms', 'µs', 'ns'):
        scaled = abs(val_s) / _ZEIT_TO_S[unit]
        if scaled >= 0.9:
            prec = ".0f" if scaled >= 100_000 else ".1f" if scaled >= 10_000 else ".2f" if scaled >= 1_000 else ".3f"
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


_EINH_STROM  = {'A', 'mA'}
_EINH_SPANNG = {'V', 'mV'}

def _produkt_einheit(einh_a: str, einh_b: str) -> str:
    """Kombinierte Einheit für Multiplikations-Integral.
    Strom × Spannung (beliebige Reihenfolge) → Spannung zuerst (VA-Konvention).
    """
    if einh_a in _EINH_STROM and einh_b in _EINH_SPANNG:
        return f"{einh_b}{einh_a}"    # A×V → VA,  mA×V → VmA,  A×mV → mVA
    if einh_a in _EINH_SPANNG and einh_b in _EINH_STROM:
        return f"{einh_a}{einh_b}"    # V×A → VA,  V×mA → VmA,  mV×A → mVA
    return f"{einh_a}·{einh_b}"


def _slider_schritt_125(spanne: float, max_schritte: int = 500) -> float:
    """Wählt aus der 1-2-5-Reihe den kleinsten Schritt, der ≤ max_schritte ergibt.

    Kandidatenreihe: …0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10…
    """
    if spanne <= 0:
        return 1.0
    mag   = 10 ** math.floor(math.log10(spanne / max_schritte))
    for m in (1, 2, 5, 10, 20, 50):
        s = mag * m
        if spanne / s <= max_schritte:
            return float(round(s, 10))
    return float(mag * 100)


_LAENGE_AUTOSCALE_STUFEN = [
    ('nm', 1e-9), ('µm', 1e-6), ('mm', 1e-3), ('m', 1.0),
]

def _fmt_val(value: float, einheit: str, prec: int = 3) -> str:
    """Formatiert Messwert mit Autoscaling für Längeneinheiten (nm/µm/mm/m).

    Für Einheiten der Form '<länge>' oder '<länge>/s' oder '<länge>/s²' wird
    der Wert automatisch in die passende Längeneinheit skaliert (10–9999).
    """
    if np.isnan(value):
        return "N/A"

    # Einheit zerlegen: Basis-Längeneinheit + Suffix (/s, /s², leer)
    _suffix = ''
    _basis = einheit
    for _sfx in ('/s²', '/s'):
        if einheit.endswith(_sfx):
            _basis = einheit[:-len(_sfx)]
            _suffix = _sfx
            break

    _laenge_si = next((f for u, f in _LAENGE_AUTOSCALE_STUFEN if u == _basis), None)

    if _laenge_si is not None and value != 0:
        # Wert in SI (m oder m/s oder m/s²)
        val_si = abs(value) * _laenge_si
        # Beste Einheit: größte bei der val_si/f >= 10
        best_u, best_f = _LAENGE_AUTOSCALE_STUFEN[0]
        for u, f in reversed(_LAENGE_AUTOSCALE_STUFEN):
            if val_si / f >= 10:
                best_u, best_f = u, f
                break
        scaled = value * (_laenge_si / best_f)
        display_einheit = f'{best_u}{_suffix}'
        abs_s = abs(scaled)
        if abs_s == 0:
            return f"0.000 {display_einheit}"
        digits_before = max(1, int(np.floor(np.log10(abs_s))) + 1)
        dec = max(0, min(prec, 6 - digits_before))
        return f"{scaled:.{dec}f} {display_einheit}"

    # Nicht-Länge oder Null: ursprüngliche Logik
    abs_v = abs(value)
    if abs_v == 0:
        return f"0.000 {einheit}"
    if abs_v < 0.001 or abs_v >= 1_000_000:
        return f"{value:.{prec}e} {einheit}"
    digits_before = max(1, int(np.floor(np.log10(abs_v))) + 1)
    dec = max(0, min(prec, 6 - digits_before))
    return f"{value:.{dec}f} {einheit}"


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
        if scaled >= 0.9:
            prec = ".0f" if scaled >= 100_000 else ".1f" if scaled >= 10_000 else ".2f" if scaled >= 1_000 else ".3f"
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


def _get_raw_bytes(f) -> bytes:
    """Gibt Datei-Bytes zurück – bei CSX-Dateien ohne den eingebetteten Settings-Block."""
    # Akzeptiere verschiedene Upload-Objekte (`UploadedFile`, BytesIO, str, bytes)
    def _ensure_bytes(obj) -> bytes:
        if obj is None:
            return b""
        if isinstance(obj, bytes):
            return obj
        if isinstance(obj, str):
            return obj.encode('utf-8')
        if hasattr(obj, 'getvalue'):
            v = obj.getvalue()
            return v if isinstance(v, bytes) else str(v).encode('utf-8')
        if hasattr(obj, 'read'):
            v = obj.read()
            return v if isinstance(v, bytes) else str(v).encode('utf-8')
        return str(obj).encode('utf-8')

    def _load_split_csx_module():
        try:
            if hasattr(reader, 'split_csx'):
                return reader
        except Exception:
            pass
        try:
            import importlib.util
            import pathlib
            rp = pathlib.Path(__file__).parent / 'reader.py'
            spec = importlib.util.spec_from_file_location('reader_local', str(rp))
            if spec is None or spec.loader is None:
                raise ImportError('Could not load reader.py spec')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, 'split_csx'):
                return module
        except Exception:
            pass
        raise AttributeError("reader module has no attribute 'split_csx'")

    fb = _ensure_bytes(f)
    split_reader = _load_split_csx_module()
    raw, _ = split_reader.split_csx(fb)
    return raw


def on_file_upload():
    """Schließt Einstellungen-Expander beim Hochladen; bei CSX werden Einstellungen geladen."""
    f = st.session_state.get('_file_uploader')
    if f is not None:
        st.session_state.einstellungen = False
        for _k in _SUB_EXPANDER_KEYS:
            st.session_state[_k] = False
        # Titel/Beschreibung bei neuer Datei zurücksetzen (CSX-Einstellungen überschreiben danach)
        if st.session_state.get('last_file_name') != f.name:
            st.session_state['pdf_titel']        = f"Messdaten-Auswertung – {f.name}"[:50]
            st.session_state['pdf_beschreibung'] = ''
        # CSX: eingebettete Einstellungen automatisch anwenden (nur beim ersten Upload)
        if (f.name.lower().endswith('.csx')
                and st.session_state.get('last_file_name') != f.name):
            try:
                fb = f.getvalue()
            except Exception:
                try:
                    fb = f.read()
                except Exception:
                    fb = f
            if isinstance(fb, str):
                fb = fb.encode('utf-8')
            try:
                _, _csx_cfg = reader.split_csx(fb)
            except AttributeError:
                # Fallback: lade reader.py direkt aus dem Arbeitsverzeichnis
                try:
                    import importlib.util
                    import pathlib
                    rp = pathlib.Path(__file__).parent / 'reader.py'
                    spec = importlib.util.spec_from_file_location('reader_local', str(rp))
                    reader_local = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(reader_local)
                    _, _csx_cfg = reader_local.split_csx(fb)
                except Exception:
                    _, _csx_cfg = None, None
            if _csx_cfg:
                for _k, _v in _csx_cfg.items():
                    if _k in EINSTELLUNGEN_KEYS:
                        st.session_state[_k] = _v
                st.session_state['window_length_sw']       = st.session_state.get('window_length', 30)
                st.session_state['window_length_accel_sw'] = st.session_state.get('window_length_accel', 40)
                st.session_state['v_time_base_s_sw']       = st.session_state.get('v_time_base_s')
                st.session_state['_csx_settings_loaded']   = True


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
        # Widget-Mirror-Keys nach Laden synchronisieren
        st.session_state['window_length_sw']       = st.session_state.get('window_length', 30)
        st.session_state['window_length_accel_sw'] = st.session_state.get('window_length_accel', 40)
        st.session_state['v_time_base_s_sw']       = st.session_state.get('v_time_base_s')
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

def get_idx_at_x(x_s: float, sample_rate: float, max_idx: int) -> int:
    """Wandelt Zeitwert in Sekunden in DataFrame-Index um. O(1)."""
    return int(np.clip(round(x_s * sample_rate), 0, max_idx))


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
    file_extensions = ["txt", "csx"]
else:
    file_extensions = ["csv", "csx"]

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
            st.toggle("Kanalnamen aus erster Zeile", key="csv_header_aus_zeile1",
                      help="Liest die erste Zeile der CSV-Datei als Kanalnamen (unabhängig von 'Kopfzeilen überspringen').")
            if st.session_state.get('csv_header_aus_zeile1', False):
                st.toggle("Kanal Einheit aus zweiter Zeile", key="csv_einheit_aus_zeile2",
                          help=f"Liest die zweite Zeile als Kanaleinheiten. Gültige Werte: {', '.join(EINHEIT_OPTIONEN)}")
        st.number_input("Max. Samples importieren", min_value=0, step=1000, key="max_samples",
                        help="Maximale Anzahl der zu importierenden Datenpunkte (0 = alle importieren).")

    _n_show = N_KANÄLE
    if uploaded_file:
        file_bytes = _get_raw_bytes(uploaded_file)
        try:
            _n_show = min(N_KANÄLE, max(1, _detect_kanal_count_cached(
                file_bytes, file_type,
                st.session_state.get('skip_rows', 12),
            )))
        except Exception:
            pass
        st.session_state['n_kanäle_datei'] = _n_show

    if uploaded_file and _n_show > 0:  # file_bytes ist hier garantiert definiert (uploaded_file truthy)
        if (file_type == "CSV plain"
                and st.session_state.get('csv_header_aus_zeile1', False)):
            _header_namen = reader.peek_csv_plain_kanalnames(file_bytes, _n_show)
            for _i, _hname in enumerate(_header_namen, 1):
                st.session_state[f'ch{_i}_name'] = _hname
            for _i in range(len(_header_namen) + 1, _n_show + 1):
                if not st.session_state.get(f'ch{_i}_name', '').strip():
                    st.session_state[f'ch{_i}_name'] = f'Kanal {_i}'
            if st.session_state.get('csv_einheit_aus_zeile2', False):
                _header_einh = reader.peek_csv_plain_einheiten(
                    file_bytes, _n_show, EINHEIT_OPTIONEN
                )
                for _i, _einh in enumerate(_header_einh, 1):
                    if _einh is not None:
                        st.session_state[f'ch{_i}_einheit'] = _einh
        else:
            for _i in range(1, _n_show + 1):
                if not st.session_state.get(f'ch{_i}_name', '').strip():
                    st.session_state[f'ch{_i}_name'] = f'Kanal {_i}'

    with st.expander("Kanäle", expanded=st.session_state.sub_kanaele, key="sub_kanaele", on_change=_SUB_EXPANDER_CBS['sub_kanaele']):
        st.caption("Leer: wird beim Schließen automatisch als 'Kanal N' benannt.")
        _osc_einheiten: list[str] = []
        if file_type == "Oszilloskop CSV" and uploaded_file:
            _, _osc_einheiten = reader.peek_oszilloskop_header(file_bytes)
        for _ki in range(N_KANÄLE):
            _i         = _ki + 1
            _ch_key    = f'ch{_i}_name'
            _einh_key  = f'ch{_i}_einheit'
            _skale_key = f'osc_skale_{_i}'
            _in_datei  = _i <= _n_show          # Kanal ist in aktueller Datei vorhanden
            _osc_hint  = f" [{_osc_einheiten[_ki]}]" if _ki < len(_osc_einheiten) else ""
            _cur_einh  = st.session_state.get(_einh_key, 'µm')
            _einh_opts = EINHEIT_OPTIONEN if _cur_einh in EINHEIT_OPTIONEN else [_cur_einh] + EINHEIT_OPTIONEN
            _ch_label  = f"Kanal {_i}{_osc_hint}" + ("" if _in_datei else " *(nicht in Datei)*")
            _col_name, _col_skale, _col_einh = st.columns([2, 1, 1])
            _col_name.text_input(
                _ch_label, key=_ch_key,
                max_chars=12,
                disabled=not _in_datei,
                help="Leer lassen um diesen Kanal nicht einzulesen." if _in_datei
                     else "Dieser Kanal ist in der aktuellen Datei nicht vorhanden. Name bleibt für nächste Datei erhalten.",
            )
            if file_type in ("Oszilloskop CSV", "CSV plain"):
                _col_skale.number_input(
                    "×",
                    step=0.01, format="%.2f", key=_skale_key,
                    disabled=not _in_datei,
                    help="Skalierungsfaktor (Rohwert × Faktor).",
                )
            _col_einh.selectbox(
                "Einheit", _einh_opts, key=_einh_key,
                disabled=not _in_datei,
                help="Physikalische Einheit – bestimmt die Y-Achse.",
            )
            if _in_datei:
                _sg_en_key  = f'ch{_i}_sg_en'
                _sg_win_key = f'ch{_i}_sg_win'
                _sg_en = st.toggle(
                    "SG-Vorfilter", key=_sg_en_key,
                    help="Savitzky-Golay-Glättung vor allen Berechnungen (Polygrad 3). Reduziert Digitalisierungsrauschen.",
                )
                if _sg_en:
                    st.slider(
                        "Fenster", 3, 99, step=2, key=_sg_win_key,
                        help="Fenstergröße des SG-Vorfilters (ungerade, 3–99). Größer = stärker geglättet.",
                    )

    if uploaded_file:
        _kanal_cfg = [st.session_state.get(f'ch{i}_name', '').strip() for i in range(1, N_KANÄLE + 1)]
        # Nur die ersten _n_show Kanäle verwenden – verhindert, dass alte Namen aus
        # früheren Dateien mit mehr Kanälen mitgeschleppt werden.
        # Duplikate entfernen (erster Treffer gewinnt) – verhindert StreamlitDuplicateElementKey
        # wenn zwei Kanäle versehentlich denselben Namen tragen.
        _sensor_ch_num: dict[str, int] = {}
        for _i, _n in enumerate(_kanal_cfg[:_n_show]):
            if _n and _n not in _sensor_ch_num:
                _sensor_ch_num[_n] = _i + 1
        kanal_namen_tuple = tuple(_sensor_ch_num.keys())

        if len(kanal_namen_tuple) < 1:
            st.sidebar.error("Mindestens ein Kanalname muss angegeben werden.")
            st.stop()

        sensor_namen = list(kanal_namen_tuple)
        # Index über _sensor_ch_num[name] statt Listenposition i+1 – sonst verschieben sich
        # die Settings, sobald ein früherer Kanal-Slot leer gelassen wird (dokumentiertes Feature).
        _osc_skale_tuple = tuple(
            st.session_state[f'osc_skale_{_sensor_ch_num[name]}'] for name in kanal_namen_tuple
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
            if file_type in ("Hubmessung", "Oszilloskop CSV"):
                total_time_s = float(df_raw['Zeit (s)'].iloc[-1])
            else:
                total_time_s = len(df_raw) / sample_rate
            # Bei CSX-Dateien Cursor/Offsets/Crop NICHT zurücksetzen – kommen aus eingebetteten Einstellungen
            if not st.session_state.pop('_csx_settings_loaded', False):
                for i in range(1, N_KANÄLE + 1):
                    st.session_state[f'off{i}']        = 0.0
                    st.session_state[f'off{i}_slider'] = 0.0
                for _i in range(1, N_KANÄLE + 1):
                    st.session_state[f'show_ch{_i}'] = True
                _setze_cursor_position(total_time_s)
                st.session_state.crop_start = None
                st.session_state.crop_end   = None
            st.session_state.zoom_token    += 1
            st.session_state.last_file_name = uploaded_file.name
            st.session_state.einstellungen  = False
            st.rerun()

    else:
        df_raw = None
        sensor_namen = []

    with st.expander("Y-Offset", expanded=st.session_state.sub_offsets, key="sub_offsets", on_change=_SUB_EXPANDER_CBS['sub_offsets']):
        if uploaded_file and df_raw is not None and sensor_namen:
            # Kanalauswahl als Radio-Buttons
            if st.session_state.get('off_kanal_sel', '') not in sensor_namen:
                st.session_state['off_kanal_sel'] = sensor_namen[0]
            _off_kanal = st.radio(
                "Kanal", sensor_namen,
                horizontal=True,
                key='off_kanal_sel',
                label_visibility="collapsed",
            )
            _off_kn = _sensor_ch_num[_off_kanal]   # tatsächliche Slot-Nummer (nicht Listenposition)

            # Zeitachse für df_raw aufbauen (Vibrometer hat keine 'Zeit (s)'-Spalte)
            _xa_off   = float(st.session_state.get('xa', 0.0))
            _xb_off   = float(st.session_state.get('xb', float('inf')))
            _hz_f_off = float(st.session_state.get('zeit_hz_faktor', 1000.0))
            if 'Zeit (s)' in df_raw.columns:
                _t_arr_off = df_raw['Zeit (s)'].values
            elif sample_rate > 0:
                _t_arr_off = np.arange(len(df_raw)) / sample_rate
            else:
                _t_arr_off = None

            # Rohsignal des gewählten Kanals im A-B-Bereich
            _sig_raw = df_raw[_off_kanal].values
            if _t_arr_off is not None and _xa_off != _xb_off:
                _t_lo    = min(_xa_off, _xb_off)
                _t_hi    = max(_xa_off, _xb_off)
                _mask_ab = (_t_arr_off >= _t_lo) & (_t_arr_off <= _t_hi)
                _sig_ab  = _sig_raw[_mask_ab] if _mask_ab.any() else _sig_raw
            else:
                _sig_ab  = _sig_raw

            _ob1, _ob2, _ob3, _ob4 = st.columns(4)
            if _ob1.button("Max.", key="off_btn_max", width="stretch",
                           help="Y-Offset: Maximalwert im A-B-Bereich → 0"):
                _v = -float(np.max(_sig_ab))
                st.session_state[f'off{_off_kn}']        = _v
                st.session_state[f'off{_off_kn}_slider'] = _v
            if _ob2.button("Avg.", key="off_btn_avg", width="stretch",
                           help="Y-Offset: Mittelwert im A-B-Bereich → 0"):
                _v = -float(np.mean(_sig_ab))
                st.session_state[f'off{_off_kn}']        = _v
                st.session_state[f'off{_off_kn}_slider'] = _v
            if _ob3.button("Min.", key="off_btn_min", width="stretch",
                           help="Y-Offset: Minimalwert im A-B-Bereich → 0"):
                _v = -float(np.min(_sig_ab))
                st.session_state[f'off{_off_kn}']        = _v
                st.session_state[f'off{_off_kn}_slider'] = _v
            if _ob4.button("↺", key="reset_offsets", width="stretch",
                           help="Setzt alle Y-Offsets auf 0."):
                for _ri in range(1, N_KANÄLE + 1):
                    st.session_state[f'off{_ri}']        = 0.0
                    st.session_state[f'off{_ri}_slider'] = 0.0

            st.segmented_control(
                "Schrittweite", _FEIN_STUFEN, key="off_fein_stufe",
                on_change=_on_fein_toggle,
                help="Schrittweite des Y-Offsets in Kanaleinheiten (z.B. µm, mm/s …).",
            )
            # Schrittweite direkt aus Auswahl (nicht abhängig vom Kanalbereich)
            _off_step = _FEIN_SCHRITT.get(st.session_state.get('off_fein_stufe', '1'), 1.0)

            # Grenzen aus Kanalbereich (numerisch konvertieren, Textreste aus Header-Zeilen ignorieren)
            _raw_col_num = pd.to_numeric(df_raw[_off_kanal], errors='coerce').dropna()
            if len(_raw_col_num) > 0:
                _off_lim = max(Y_OFFSET_LIMIT_MIN,
                               abs(float(_raw_col_num.min())) * Y_OFFSET_LIMIT_FAKTOR,
                               abs(float(_raw_col_num.max())) * Y_OFFSET_LIMIT_FAKTOR)
            else:
                _off_lim = Y_OFFSET_LIMIT_MIN
            _off_lim  = float(np.ceil(_off_lim / 100) * 100)
            _off_fmt  = "%.2f"
            # Freien Key klemmen – beide Keys konsistent halten (vor Widget-Render)
            _off_cur = float(np.clip(
                float(st.session_state.get(f'off{_off_kn}', 0.0)),
                -_off_lim, _off_lim,
            ))
            st.session_state[f'off{_off_kn}']        = _off_cur
            st.session_state[f'off{_off_kn}_slider'] = _off_cur
            st.number_input(
                _off_kanal,
                min_value=-_off_lim, max_value=_off_lim,
                step=_off_step, format=_off_fmt,
                key=f'off{_off_kn}_slider', on_change=OFF_CALLBACKS[_off_kn - 1],
                label_visibility="collapsed",
            )

    with st.expander("X-Offset", expanded=st.session_state.sub_xoffset, key="sub_xoffset", on_change=_SUB_EXPANDER_CBS['sub_xoffset']):
        if uploaded_file and sensor_namen:
            st.segmented_control(
                "Schrittweite", _FEIN_STUFEN, key="x_off_fein_stufe",
                help=f"Schrittweite des X-Offsets in {_zeit_einheit}.",
            )
            _x_off_step = _FEIN_SCHRITT.get(st.session_state.get('x_off_fein_stufe', '0,1'), 0.1)
            for _xname in sensor_namen:
                st.number_input(
                    f"X-Offset {_xname} ({_zeit_einheit})",
                    step=_x_off_step, format="%.3f",
                    key=f'x_off{_sensor_ch_num[_xname]}',
                    help=f"Zeitversatz in {_zeit_einheit} – verschiebt diesen Kanal nach links (−) oder rechts (+).",
                )

    with st.expander("Diagramm-Grenzwerte", expanded=st.session_state.sub_grenzwerte, key="sub_grenzwerte", on_change=_SUB_EXPANDER_CBS['sub_grenzwerte']):
        st.toggle("Gleiche Nulllinie", key="gleiche_nulllinie",
                  help="Alle Kanäle auf Automatik werden so skaliert, dass die 0-Linie auf gleicher Höhe liegt.")
        st.caption("min / max  (0 = automatisch)")
        _grenz_hat_inhalt = False
        for _i in range(1, N_KANÄLE + 1):
            _ch_name = st.session_state.get(f'ch{_i}_name', '').strip()
            if not _ch_name:
                continue
            if not st.session_state.get(f'show_ch{_i}', True):
                continue
            _ch_einh = st.session_state.get(f'ch{_i}_einheit', 'µm')
            st.caption(f"{_ch_name}  ({_ch_einh})")
            _gc1, _gc2 = st.columns(2)
            _gc1.number_input(f"min {_ch_name}", step=1.0, format="%.2f",
                              key=f'ch{_i}_ymin', label_visibility="collapsed")
            _gc2.number_input(f"max {_ch_name}", step=1.0, format="%.2f",
                              key=f'ch{_i}_ymax', label_visibility="collapsed")
            _grenz_hat_inhalt = True
        if st.session_state.get('show_velocity', False):
            st.caption("Geschwindigkeit (v)")
            _vc1, _vc2 = st.columns(2)
            _vc1.number_input("min v", step=100.0, format="%.0f",
                              key="v_axis_min", label_visibility="collapsed")
            _vc2.number_input("max v", step=100.0, format="%.0f",
                              key="v_axis_max", label_visibility="collapsed")
            _grenz_hat_inhalt = True
        if st.session_state.get('show_acceleration', False):
            st.caption("Beschleunigung (a)")
            _ac1, _ac2 = st.columns(2)
            _ac1.number_input("min a", step=500.0, format="%.0f",
                              key="a_axis_min", label_visibility="collapsed")
            _ac2.number_input("max a", step=500.0, format="%.0f",
                              key="a_axis_max", label_visibility="collapsed")
            _grenz_hat_inhalt = True
        if st.session_state.get('show_integral_curve', False):
            st.caption("∫IN1 dt")
            _ic1, _ic2 = st.columns(2)
            _ic1.number_input("min ∫IN1 dt", step=0.001, format="%.4f",
                              key="int_axis_min", label_visibility="collapsed")
            _ic2.number_input("max ∫IN1 dt", step=0.001, format="%.4f",
                              key="int_axis_max", label_visibility="collapsed")
            _grenz_hat_inhalt = True
        _mc_sel = st.session_state.get('multi_kanal_auswahl', [])
        if st.session_state.get('show_multi_diag', False) and len(_mc_sel) == 2:
            st.caption(f"∫{_mc_sel[0]}×{_mc_sel[1]} dt")
            _mc1, _mc2 = st.columns(2)
            _mc1.number_input(f"min ∫{_mc_sel[0]}×{_mc_sel[1]} dt", step=0.001, format="%.4f",
                              key="multi_diag_ymin", label_visibility="collapsed")
            _mc2.number_input(f"max ∫{_mc_sel[0]}×{_mc_sel[1]} dt", step=0.001, format="%.4f",
                              key="multi_diag_ymax", label_visibility="collapsed")
            _grenz_hat_inhalt = True
        if not _grenz_hat_inhalt:
            st.caption("Keine aktiven Kanäle oder Ableitungen.")

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
            width="stretch",
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

# _sensor_ch_num nutzen statt _kanal_cfg iterieren – verhindert, dass ein doppelter Kanalname
# aus einem alten Session-State-Slot die Einheit des aktiven Kanals überschreibt.
kanal_einheit_map = {
    nm: st.session_state.get(f'ch{slot}_einheit', 'µm')
    for nm, slot in _sensor_ch_num.items()
}

if len(kanal_namen_tuple) < 1:
    st.sidebar.error("Mindestens ein Kanalname muss angegeben sein.")
    st.stop()

# ---------------------------------------------------------------------------
# DATENAUFBEREITUNG
# ---------------------------------------------------------------------------

# Index über _sensor_ch_num[name] statt Listenposition i+1 – siehe Kommentar bei _osc_skale_tuple.
offs = tuple(float(st.session_state.get(f'off{_sensor_ch_num[name]}', 0.0)) for name in sensor_namen)

_hz_faktor    = st.session_state.get('zeit_hz_faktor', 1000.0)
_zeit_einheit = _ZEIT_HZ_ZU_EINHEIT.get(round(_hz_faktor), 'ms')

# SG-VORFILTER auf Rohkanäle – VOR Y-Offset, damit der Offset auf geglätteten Werten liegt
# Index über _sensor_ch_num[name] statt Listenposition – siehe Kommentar bei _osc_skale_tuple.
_sg_params = tuple(
    (bool(st.session_state.get(f'ch{_sensor_ch_num[name]}_sg_en', False)),
     int(st.session_state.get(f'ch{_sensor_ch_num[name]}_sg_win', 5)))
    for name in sensor_namen
)
_df_raw_filt = _apply_sg_vorfilter_cached(df_raw, kanal_namen_tuple, _sg_params)

df_full, sample_rate = _build_display_df_cached(
    _df_raw_filt, file_type, sample_rate, kanal_namen_tuple, offs,
)

# ---------------------------------------------------------------------------
# X-OFFSET – als Sample-Shift einbauen
# RAW → [scale + Y-offset] = df_full  →  [X-shift] = df_use  →  [Crop] = df
# ---------------------------------------------------------------------------

# Index über _sensor_ch_num[name] statt Listenposition – siehe Kommentar bei _osc_skale_tuple.
_x_offs = [float(st.session_state.get(f'x_off{_sensor_ch_num[name]}', 0.0)) for name in sensor_namen]
if any(v != 0.0 for v in _x_offs) and len(df_full) > 1:
    _dt_s = float(df_full['Zeit (s)'].iloc[1] - df_full['Zeit (s)'].iloc[0])
    df_use = df_full.copy()
    for _si, _sname in enumerate(sensor_namen):
        _x_off_s = _x_offs[_si] * _ZEIT_TO_S.get(_zeit_einheit, 1e-3)   # Anzeigeeinheit → s
        _n = int(round(_x_off_s / _dt_s)) if _dt_s > 0 else 0
        if _n != 0:
            _col = np.roll(df_use[_sname].values, _n)
            if _n > 0:
                _col[:_n] = np.nan
            else:
                _col[_n:] = np.nan
            df_use[_sname] = _col
else:
    df_use = df_full

max_zeit_full = float(df_use['Zeit (s)'].iloc[-1])
max_idx_full  = len(df_use) - 1

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
    df       = df_use.iloc[ci_start:ci_end + 1].reset_index(drop=True)
    min_zeit = float(df['Zeit (s)'].iloc[0])
    max_zeit = float(df['Zeit (s)'].iloc[-1])
    max_idx  = len(df) - 1
else:
    df       = df_use
    min_zeit = 0.0
    max_zeit = max_zeit_full
    max_idx  = max_idx_full

# Automatische Einheitenskalierung: feinste Einheit wählen, bei der der
# sichtbare Zeitbereich ohne viele Nachkommastellen dargestellt werden kann.
# Beispiel: 200 µs crop bei ms-Ladedatei → Anzeige in µs statt 0,2 ms.
_vis_range_s  = max_zeit - min_zeit   # bereits in Sekunden
_vis_hz, _vis_einheit = (reader.wahl_zeiteinheit(_vis_range_s)
                          if _vis_range_s > 0 else (_hz_faktor, _zeit_einheit))
_vis_scale = _vis_hz   # Sekunden × _vis_hz = Anzeigewert (z.B. × 1000 → ms)
st.session_state['_vis_scale'] = _vis_scale

# ---------------------------------------------------------------------------
# SIDEBAR: AUSWERTUNGS-STEUERUNG
# ---------------------------------------------------------------------------

st.sidebar.header("2. Auswertung")
if st.session_state.get('active_sensor_name', '') not in sensor_namen:
    st.session_state['active_sensor_name'] = sensor_namen[0]
active_sensor = st.sidebar.radio(
    "Kanal für Messung:", sensor_namen,
    horizontal=True, label_visibility="collapsed",
    key='active_sensor_name',
    help="Aktiver Kanal (IN1) für alle Berechnungen: Cursor-Messung, d/dt-max, d²/dt²-max und SOP.",
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

xa = round(float(np.clip(st.session_state.xa, min_zeit, max_zeit)), 9)
xb = round(float(np.clip(st.session_state.xb, min_zeit, max_zeit)), 9)
st.session_state.xa = xa
st.session_state.xb = xb

# Relative Zeitachse: Nullpunkt = eingefrorener Offset in Sekunden (gesetzt beim Aktivieren von Start@0).
t_offset = float(st.session_state.get('x_rel_offset', 0.0)) if st.session_state.get('x_rel_mode', False) else 0.0
st.session_state['_t_offset'] = t_offset   # Callbacks im nächsten Run lesen diesen Wert

# xa_sw/xb_sw in Anzeigeeinheit VOR Widget-Rendering setzen.
# Sekunden → Anzeigeeinheit: × _vis_scale (= _vis_hz, z.B. 1000 für ms)
st.session_state.xa_sw = round((xa - t_offset) * _vis_scale, 6)
st.session_state.xb_sw = round((xb - t_offset) * _vis_scale, 6)
# Berechnungen nutzen immer den sortierten Bereich – Slider-Reihenfolge egal
if xa > xb:
    xa, xb = xb, xa

with st.sidebar.expander("Diagrammarker", expanded=False):
    st.caption("Peak-Marker", help="Sichtbarkeit der Peak-Marker im Diagramm.")
    mark_vmax  = st.toggle("D-max",  key="mark_vmax",  help="Linie des höchsten Geschwindigkeits-Peaks (betragsmäßig) einzeichnen.")
    mark_vmin  = st.toggle("D-min",  key="mark_vmin",  help="Linie des stärksten Abfalls (negativster Geschwindigkeitspeak) einzeichnen.")
    mark_afall = st.toggle("D2-max", key="mark_afall", help="Marker des größten Beschleunigungs-Peaks (fallende Flanke) einzeichnen.")
    mark_arise = st.toggle("D2-min", key="mark_arise", help="Marker des negativsten Beschleunigungs-Peaks (steigende Flanke) einzeichnen.")
    _x_len = max(1e-9, max_zeit - min_zeit)   # Sekunden
    # Bereich: 0,5 % … 50 % der Diagrammzeit (in Sekunden)
    _tb_min_curr = _x_len * 0.005
    _tb_max_curr = _x_len * 0.50
    # Anzeigeeinheit für den Slider autoscale (Sekunden → passende Einheit)
    _tb_hz, _tb_s_einheit = reader.wahl_zeiteinheit(_tb_max_curr)
    st.session_state['_tb_hz'] = _tb_hz   # Callback liest den Faktor für die Rückrechnung in Sekunden
    # Float64 mit max. Auflösung – mind. 9 Nachkommastellen für µs-Bereich bei kleinen Crops.
    # np.float64 hat ~15 signifikante Stellen, daher reicht das auch bei kleinen Werten.
    _tb_min_d = float(np.float64(_tb_min_curr * _tb_hz))
    _tb_max_d = float(np.float64(_tb_max_curr * _tb_hz))
    # Schrittweite fest auf 200 Schritte innerhalb des gecropten Zeitbereichs.
    # float64 garantiert ausreichende Auflösung (~15 Stellen), auch bei µs-Crops.
    _tb_step_d = float(np.float64((_tb_max_d - _tb_min_d) / 200.0))
    _tb_dez = max(0, -int(math.floor(math.log10(_tb_step_d)))) if _tb_step_d >= 1e-12 else 2
    _tb_fmt = f"%.{_tb_dez}f"
    # Gespeicherten s-Wert in Slider-Einheit umrechnen; fehlt er → autom. Default (3% der Zeitachse)
    _tb_saved_s = st.session_state.get('v_time_base_s')
    if _tb_saved_s is not None and _tb_min_curr <= _tb_saved_s <= _tb_max_curr:
        # Gespeicherter Wert liegt im gültigen Bereich → übernehmen (float64)
        _tb_def_d = float(np.float64(_tb_saved_s * _tb_hz))
    else:
        # Wert fehlt oder ist außerhalb des neuen Bereichs (z.B. nach Crop-Änderung
        # mit kleinerem Zeitfenster) → Default: 3% der aktuellen Zeitachse.
        # Verhindert, dass der Slider auf sein Maximum springt.
        _tb_def_d = float(np.clip(np.float64(_x_len * 0.03 * _tb_hz), _tb_min_d, _tb_max_d))
    # Widget-Mirror-Key immer auf geclippten Wert setzen – verhindert Out-of-Range nach Datei-Wechsel
    st.session_state['v_time_base_s_sw'] = _tb_def_d
    _v_tb_display = st.slider(
        f"Zeitfenster d/dt min./max. ({_tb_s_einheit})",
        _tb_min_d, _tb_max_d,
        step=_tb_step_d, format=f"{_tb_fmt} {_tb_s_einheit}",
        key='v_time_base_s_sw',
        on_change=_on_v_time_base_cb,
        help="Mittelungsfenster für d/dt-max, d²/dt²-max und SOP: Der Peak wird über dieses Zeitfenster gemittelt. Kleiner = empfindlicher, größer = robuster gegenüber Rauschen.",
    )
    # Slider-Einheit → Sekunden mit float64-Auflösung (mind. 9 Nachkommastellen)
    v_time_base_s = float(np.float64(_v_tb_display) / np.float64(_tb_hz))
    st.session_state['v_time_base_s'] = v_time_base_s
    st.divider()
    st.caption("Diagrammlinien")
    show_v_avg       = st.toggle("Schnittlinie A–B", key="show_v_avg",
                                 help="Verbindungslinie von XA nach XB – zeigt die mittlere Änderungsrate des aktiven Kanals zwischen A und B.")
    show_rect_fit_top = st.toggle("Rechteck-Fit Top/Bot.", key="show_rect_fit_top",
                                  help="Obere und untere gestrichelte Linie des Rechteck-Fits einzeichnen.")
    show_rect_fit    = st.toggle("Rechteck-Fit füllen", key="show_rect_fit",
                                 help="Vertikale Kantenlinien und hellgrüne Füllung für alle erkannten Rechteck-Pulse.")
    show_y_slider = st.toggle("Y-Slider", key="show_y_slider",
                               help="Zwei horizontale Hilfslinien (YA / YB) frei im Diagramm positionierbar.")
    if show_y_slider and df_raw is not None and sensor_namen:
        # Y-Bereich aus gespeichertem Y-Achsen-Skalenbereich des aktiven Kanals (vorheriger Renderdurchlauf)
        _ys_lo = float(st.session_state.get('_ys_axis_lo', 0.0))
        _ys_hi = float(st.session_state.get('_ys_axis_hi', 0.0))
        if _ys_lo == _ys_hi:
            # Fallback wenn noch kein Skalenbereich gespeichert (erster Lauf)
            _ys_col  = df_use[active_sensor] if active_sensor in df_use.columns else df_use.iloc[:, 0]
            _ys_lo   = float(_ys_col.min())
            _ys_hi   = float(_ys_col.max())
            _ys_span = _ys_hi - _ys_lo
            _ys_hi  += _ys_span * Y_PUFFER
        if _ys_lo == _ys_hi:
            _ys_hi = _ys_lo + 1.0
        _ys_step  = _slider_schritt_125(_ys_hi - _ys_lo)
        _ys_dez   = max(0, -int(np.floor(np.log10(_ys_step)))) if _ys_step < 1 else 0
        _ys_fmt   = f"%.{_ys_dez}f"
        _ya_val   = float(np.clip(st.session_state.get('ya_sw', _ys_lo), _ys_lo, _ys_hi))
        _yb_val   = float(np.clip(st.session_state.get('yb_sw', _ys_hi), _ys_lo, _ys_hi))
        st.session_state['ya_sw'] = _ya_val
        st.session_state['yb_sw'] = _yb_val
        st.slider("YA", _ys_lo, _ys_hi, step=_ys_step, format=_ys_fmt,
                  key="ya_sw", label_visibility="collapsed",
                  help="Horizontale Hilfslinie YA (untere Referenz).")
        st.slider("YB", _ys_lo, _ys_hi, step=_ys_step, format=_ys_fmt,
                  key="yb_sw", label_visibility="collapsed",
                  help="Horizontale Hilfslinie YB (obere Referenz).")

show_velocity = st.sidebar.toggle(
    f"d {active_sensor} /dt anzeigen", key="show_velocity",
    help="Zeigt die 1. Ableitung (Geschwindigkeit) des aktiven Kanals auf einer zweiten Y-Achse.",
)
if show_velocity:
    st.sidebar.slider(
        "Glättung D", 5, 400, step=1,
        key="window_length_sw", on_change=_on_window_length_cb,
        help="Fenstergröße des Savitzky-Golay-Filters für die 1. Ableitung. Größer = glatter, aber geringere Detailauflösung.",
    )
show_acceleration = st.sidebar.toggle(
    f"d²{active_sensor}/dt² anzeigen", key="show_acceleration",
    help="Zeigt die 2. Ableitung (Beschleunigung) des aktiven Kanals auf einer dritten Y-Achse.",
)
if show_acceleration:
    st.sidebar.slider(
        "Glättung D2", 10, 400, step=1,
        key="window_length_accel_sw", on_change=_on_window_length_accel_cb,
        help="Fenstergröße des Savitzky-Golay-Filters für die 2. Ableitung. Größere Werte nötig, da die 2. Ableitung stärker rauscht.",
    )
show_sop = st.sidebar.toggle(
    "Speed on Point (SOP)", key="show_sop",
    help="Zeigt die Geschwindigkeit an der steigenden Flanke des Rechtecksignals an einem einstellbaren Hub. Erfordert erkanntes Rechteck-Fit.",
)
if show_sop:
    st.sidebar.slider(
        "SOP Pegel (%)", 0, 100, step=1,
        key="sop_percent",
        help="Höhe auf der steigenden Flanke in Prozent des Hub (0 % = unterer Pegel, 100 % = oberer Pegel).",
    )

show_integral = st.sidebar.toggle(
    f"∫{active_sensor} dt im Diagramm", key="show_integral",
    help="Zeichnet die Fläche unter dem aktiven Kanal zwischen XA und XB transparent ein (Integralbereich).",
)
show_integral_curve = False
if show_integral:
    show_integral_curve = st.sidebar.toggle(
        f"∫{active_sensor} dt Verlauf", key="show_integral_curve",
        help="Zeichnet den kumulativen Integralverlauf zwischen XA und XB auf einer eigenen Y-Achse (Startwert 0 bei XA).",
    )

show_multi_kanal = st.sidebar.toggle(
    "∫ IN1×IN2 dt (A-B)", key="show_multi_kanal",
    help="Multipliziert zwei Kanäle und integriert das Produkt zwischen XA und XB. Anwendung z. B.: Strom × Spannung → VA·s.",
)
if show_multi_kanal:
    # Ungültige Vorauswahl (z. B. aus vorheriger Datei) bereinigen
    _valid_multi = [c for c in st.session_state.get('multi_kanal_auswahl', []) if c in sensor_namen]
    if _valid_multi != list(st.session_state.get('multi_kanal_auswahl', [])):
        st.session_state['multi_kanal_auswahl'] = _valid_multi
    st.sidebar.multiselect(
        "Kanäle wählen (genau 2)",
        options=sensor_namen,
        key="multi_kanal_auswahl",
        help="2 Kanäle für die Multiplikation auswählen.",
    )
    if len(st.session_state.get('multi_kanal_auswahl', [])) == 2:
        _mc_names = st.session_state['multi_kanal_auswahl']
        st.sidebar.toggle(
            f"∫{_mc_names[0]}×{_mc_names[1]} dt im Diagramm",
            key="show_multi_diag",
            help="Zeichnet die Produkt-Kurve (IN1×IN2) und den Integralbereich zwischen XA und XB im Diagramm.",
        )
show_multi_diag = st.session_state.get('show_multi_diag', False)

_SPANNUNG_ZU_V = {'V': 1.0, 'mV': 1e-3}
show_widerstand_integral = st.sidebar.toggle(
    "∫U/R dt (A-B)", key="show_widerstand_integral",
    help="Integriert den berechneten Strom I=U/R zwischen XA und XB. Spannungskanal wählen und Widerstand angeben.",
)
if show_widerstand_integral:
    _u_kanäle = [n for n in sensor_namen if kanal_einheit_map.get(n, '') in _SPANNUNG_ZU_V]
    if _u_kanäle:
        if st.session_state.get('widerstand_kanal', '') not in _u_kanäle:
            st.session_state['widerstand_kanal'] = _u_kanäle[0]
        st.sidebar.selectbox(
            "Spannungskanal",
            options=_u_kanäle,
            key="widerstand_kanal",
            help="Kanal mit Spannungseinheit (V/mV) für I = U / R.",
        )
        st.sidebar.number_input(
            "Widerstand (kΩ)",
            min_value=0.001,
            step=1.0,
            format="%.3f",
            key="widerstand_kohm",
            help="Widerstandswert in kΩ. Standard: 200 kΩ.",
        )
    else:
        st.sidebar.caption("Kein Kanal mit Spannungseinheit (V/mV) verfügbar.")

rect_fit = compute_best_fit_rectangle(
    df_use['Zeit (s)'].values,
    df_use[active_sensor].values,
)

# ---------------------------------------------------------------------------
# MESSWERTBERECHNUNG
# ---------------------------------------------------------------------------

if crop_active:
    idx_a = get_idx_at_x(xa - min_zeit, sample_rate, max_idx)
    idx_b = get_idx_at_x(xb - min_zeit, sample_rate, max_idx)
else:
    idx_a = get_idx_at_x(xa, sample_rate, max_idx)
    idx_b = get_idx_at_x(xb, sample_rate, max_idx)

ya = df.loc[idx_a, active_sensor]
yb = df.loc[idx_b, active_sensor]

dt_val_s = abs(xb - xa)   # Sekunden
dy       = abs(yb - ya)
v_avg    = dy / dt_val_s * v_faktor if dt_val_s > 0 else 0.0

halbes_zeitfenster = max(1, int(v_time_base_s * sample_rate / 2))

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
    _t_int = df.loc[idx_start:idx_end, 'Zeit (s)'].values   # Sekunden
    _y_int = df.loc[idx_start:idx_end, active_sensor].values
    integral_val = float(np.trapezoid(_y_int, _t_int))   # [aktiv_einheit × s]
else:
    integral_val = 0.0

# Kanal-Multiplikation und Integration zwischen XA und XB (Trapezregel, volle Auflösung)
multi_integral_val  = float('nan')
_multi_auswahl      = st.session_state.get('multi_kanal_auswahl', [])
if show_multi_kanal and len(_multi_auswahl) == 2:
    _mc_a, _mc_b = _multi_auswahl
    if _mc_a in df.columns and _mc_b in df.columns and idx_end > idx_start:
        _t_mi   = df.loc[idx_start:idx_end, 'Zeit (s)'].values   # Sekunden
        _y_mi   = df.loc[idx_start:idx_end, _mc_a].values * df.loc[idx_start:idx_end, _mc_b].values
        multi_integral_val = float(np.trapezoid(_y_mi, _t_mi))  # [einh_A × einh_B × s]

# Widerstands-Integral: ∫(U/R) dt zwischen XA und XB (Trapezregel, volle Auflösung)
widerstand_integral_val = float('nan')
_w_kanal = st.session_state.get('widerstand_kanal', '')
_w_kohm  = float(st.session_state.get('widerstand_kohm', 200.0))
if show_widerstand_integral and _w_kanal in df.columns and _w_kohm > 0 and idx_end > idx_start:
    _w_einh   = kanal_einheit_map.get(_w_kanal, 'V')
    _u_zu_v   = _SPANNUNG_ZU_V.get(_w_einh, 1.0)
    _r_ohm    = _w_kohm * 1_000.0
    _t_wi     = df.loc[idx_start:idx_end, 'Zeit (s)'].values   # Sekunden
    _u_wi     = df.loc[idx_start:idx_end, _w_kanal].values
    _i_wi     = (_u_wi * _u_zu_v) / _r_ohm           # A
    widerstand_integral_val = float(np.trapezoid(_i_wi, _t_wi))  # [A × s]

# ---------------------------------------------------------------------------
# SG-ABLEITUNGEN – einmalig auf vollem Datensatz (Messwerte + Diagramm)
# ---------------------------------------------------------------------------

arr_full  = df_use[active_sensor].values   # immer voller Datensatz – verhindert SG-Randeffekte beim Crop
dt_step_s = 1.0 / sample_rate
sg_v_roh_full = _sg_ableitung_cached(arr_full, dt_step_s, st.session_state.window_length, 1)
sg_a_roh_full = _sg_ableitung_cached(arr_full, dt_step_s, st.session_state.window_length_accel, 2)
# Gecropten Slice extrahieren damit alle Folgeberechnungen auf df-Indizes passen
if crop_active and sg_v_roh_full is not None:
    sg_v_roh = sg_v_roh_full[ci_start:ci_end + 1]
    sg_a_roh = sg_a_roh_full[ci_start:ci_end + 1] if sg_a_roh_full is not None else None
else:
    sg_v_roh = sg_v_roh_full
    sg_a_roh = sg_a_roh_full

# Peak-Marker-Initialisierung (werden nur gesetzt wenn genug Datenpunkte vorhanden)
t_vmax_start, y_vmax_start = None, None
t_vmax_ende,  y_vmax_ende  = None, None
t_vmin_start, y_vmin_start = None, None
t_vmin_ende,  y_vmin_ende  = None, None
t_amax_falling, y_amax_falling = None, None
t_amax_rising,  y_amax_rising  = None, None
has_vmax         = False
has_vmin         = False
has_amax_falling = False
has_amax_rising  = False
v_max            = float('nan')
v_min            = float('nan')
a_max_falling    = float('nan')
a_min_rising     = float('nan')
sop_linien: list = []
v_sop            = float('nan')

if idx_end > idx_start:
    if sg_v_roh is not None:
        sg_v_signed = sg_v_roh * v_faktor

        # D-max: höchste (positivste) Geschwindigkeit – vorzeichenkorrekt
        sg_v_signed_slice = sg_v_signed[idx_start:idx_end + 1]
        idx_vmax_peak_loc = int(np.argmax(sg_v_signed_slice))
        idx_vmax_peak     = idx_start + idx_vmax_peak_loc
        iv_start          = max(0, idx_vmax_peak - halbes_zeitfenster)
        iv_ende           = min(max_idx, idx_vmax_peak + halbes_zeitfenster)
        v_max             = float(np.mean(sg_v_signed[iv_start:iv_ende + 1]))
        if 0 <= iv_start <= max_idx and 0 <= iv_ende <= max_idx:
            t_vmax_start = df.loc[iv_start, 'Zeit (s)']
            y_vmax_start = df.loc[iv_start, active_sensor]
            t_vmax_ende  = df.loc[iv_ende,  'Zeit (s)']
            y_vmax_ende  = df.loc[iv_ende,  active_sensor]
            has_vmax     = True

        # D-min: negativster vorzeichenbehafteter Peak
        idx_vmin_peak_loc  = int(np.argmin(sg_v_signed_slice))
        idx_vmin_peak      = idx_start + idx_vmin_peak_loc
        iv_min_start       = max(0, idx_vmin_peak - halbes_zeitfenster)
        iv_min_ende        = min(max_idx, idx_vmin_peak + halbes_zeitfenster)
        v_min              = float(np.mean(sg_v_signed[iv_min_start:iv_min_ende + 1]))
        if 0 <= iv_min_start <= max_idx and 0 <= iv_min_ende <= max_idx:
            t_vmin_start = df.loc[iv_min_start, 'Zeit (s)']
            y_vmin_start = df.loc[iv_min_start, active_sensor]
            t_vmin_ende  = df.loc[iv_min_ende,  'Zeit (s)']
            y_vmin_ende  = df.loc[iv_min_ende,  active_sensor]
            has_vmin     = True

    if sg_a_roh is not None:
        sg_a = sg_a_roh * a_faktor

        def _peak_marker(idx_abs):
            """Gemittelter Beschleunigungswert und Diagramm-Position für einen Peak."""
            ia0  = max(idx_start, idx_abs - halbes_zeitfenster)
            ia1  = min(idx_end,   idx_abs + halbes_zeitfenster)
            wert = float(np.mean(sg_a[ia0:ia1 + 1]))
            return wert, float(df.loc[idx_abs, 'Zeit (s)']), float(df.loc[idx_abs, active_sensor])

        sg_a_slice = sg_a[idx_start:idx_end + 1]

        idx_falling_abs                                = idx_start + int(np.argmax(sg_a_slice))
        a_max_falling, t_amax_falling, y_amax_falling = _peak_marker(idx_falling_abs)
        has_amax_falling = True

        idx_rising_abs                               = idx_start + int(np.argmin(sg_a_slice))
        a_min_rising, t_amax_rising, y_amax_rising   = _peak_marker(idx_rising_abs)
        has_amax_rising = True

# --- Längen-Autoscale: v/a Einheit auf 10–9999 Bereich anpassen ---
# ZWEIPHASIG: _ableit_info liefert v_faktor=1.0 für Längeneinheiten (Rohwert in einheit/s).
# Dieser Block wählt die beste Anzeigeeinheit (nm/µm/mm/m) und überschreibt v_faktor/a_faktor
# mit dem Skalierungsfaktor. Alle Folgenutzer (SOP, Plotly-Traces, Export)
# erhalten damit direkt anzeigerichtige Werte – ohne zweiten Multiplikationsschritt.
if _aktiv_einheit in _LAENGE_EINHEITEN:
    # Referenzwert in kanal_einheit/s (v_faktor=1 für Längen, Werte direkt nutzbar)
    _v_ref = max(abs(v_max) if not np.isnan(v_max) else 0.0,
                 abs(v_min) if not np.isnan(v_min) else 0.0)
    _a_ref = max(abs(a_max_falling) if not np.isnan(a_max_falling) else 0.0,
                 abs(a_min_rising)  if not np.isnan(a_min_rising)  else 0.0)
    if _v_ref > 0:
        _v_disp_eu, _v_scale = _laenge_autoscale(_aktiv_einheit, _v_ref)
        v_einheit     = f'{_v_disp_eu}/s'
        v_faktor      = _v_scale          # SG-Rohwert → angezeigte Einheit/s
        v_max         = v_max         * _v_scale if not np.isnan(v_max)  else v_max
        v_min         = v_min         * _v_scale if not np.isnan(v_min)  else v_min
        # v_avg und v_cursor_delta wurden mit v_faktor=1.0 in einheit/s berechnet
        v_avg          = v_avg          * _v_scale
        v_cursor_delta = v_cursor_delta * _v_scale if not np.isnan(v_cursor_delta) else v_cursor_delta
        v_at_xa        = v_at_xa        * _v_scale if not np.isnan(v_at_xa)        else v_at_xa
        v_at_xb        = v_at_xb        * _v_scale if not np.isnan(v_at_xb)        else v_at_xb
    if _a_ref > 0:
        _a_disp_eu, _a_scale = _laenge_autoscale(_aktiv_einheit, _a_ref)
        a_einheit     = f'{_a_disp_eu}/s²'
        a_faktor      = _a_scale           # SG-Rohwert (einheit/s²) → angezeigte Einheit/s² (linear, /s² steckt bereits in delta=dt_s der SG-Ableitung)
        a_max_falling = a_max_falling * _a_scale if not np.isnan(a_max_falling) else a_max_falling
        a_min_rising  = a_min_rising  * _a_scale if not np.isnan(a_min_rising)  else a_min_rising

if show_sop and rect_fit is not None:
    # sg_v_roh_full (volle Länge = df_use) verwenden, nicht das crop-geslicte sg_v_roh –
    # _finde_sop_kreuzungen und rect_fit arbeiten auf dem df_use-Index-Raum.
    _sg_v_sop = (sg_v_roh_full * v_faktor) if sg_v_roh_full is not None else None
    sop_linien, v_sop = _finde_sop_kreuzungen(
        df_use['Zeit (s)'].values,
        df_use[active_sensor].values,
        rect_fit,
        st.session_state.sop_percent,
        sample_rate,
        halbes_zeitfenster,
        v_faktor=v_faktor,
        sg_v=_sg_v_sop,
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
_y_span_plot = y_max_plot - y_min_plot
if _y_span_plot == 0:  # Flaches Signal → Plotly würde Bereich ignorieren ohne Mindest-Span
    _y_span_plot = max(abs(y_max_plot) * 0.1, 1.0)
y_range_plot = [y_min_plot - _y_span_plot * 0.05, y_max_plot + _y_span_plot * Y_PUFFER]

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

# --- Gleiche Nulllinie: interne Limits berechnen (alle Automatik-Achsen) ---
_nulllinie_ranges: dict[str, tuple[float, float]] = {}
if st.session_state.get('gleiche_nulllinie', False) and len(sichtbare_sensor_namen) > 1:
    def _runde_scale(v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            return 1.0
        mag = 10 ** math.floor(math.log10(v))
        for fak in (1, 2, 5, 10, 20, 50, 100):
            k = fak * mag
            if k >= v:
                return k
        return v

    # Alle Automatik-Achsen: reguläre Kanäle + Zusatzachsen (D, D², Integral, Multi-Diag)
    # Schlüssel: Kanalname (reguläre Kanäle) oder Sonderschlüssel __v__, __a__, __int__, __md__
    _alle_auto_bereiche: dict[str, tuple[float, float]] = {}

    for _n in sichtbare_sensor_namen:
        _ci = _sensor_ch_num.get(_n, 0)
        _lo = float(st.session_state.get(f'ch{_ci}_ymin', 0))
        _hi = float(st.session_state.get(f'ch{_ci}_ymax', 0))
        if _lo == 0 and _hi == 0 and _n in _kanal_bereiche:
            _alle_auto_bereiche[_n] = _kanal_bereiche[_n]

    if show_velocity and sg_v_roh is not None:
        if float(st.session_state.get('v_axis_min', 0)) == 0 and float(st.session_state.get('v_axis_max', 0)) == 0:
            _v_arr = sg_v_roh * v_faktor
            _v_lo_nl, _v_hi_nl = float(np.nanmin(_v_arr)), float(np.nanmax(_v_arr))
            if math.isfinite(_v_lo_nl) and math.isfinite(_v_hi_nl):
                _alle_auto_bereiche['__v__'] = (_v_lo_nl, _v_hi_nl)

    if show_acceleration and sg_a_roh is not None:
        if float(st.session_state.get('a_axis_min', 0)) == 0 and float(st.session_state.get('a_axis_max', 0)) == 0:
            _a_arr = sg_a_roh * a_faktor
            _a_lo_nl, _a_hi_nl = float(np.nanmin(_a_arr)), float(np.nanmax(_a_arr))
            if math.isfinite(_a_lo_nl) and math.isfinite(_a_hi_nl):
                _alle_auto_bereiche['__a__'] = (_a_lo_nl, _a_hi_nl)

    if show_integral_curve and active_sensor in df_use.columns and len(df_use) > 1:
        if float(st.session_state.get('int_axis_min', 0)) == 0 and float(st.session_state.get('int_axis_max', 0)) == 0:
            _t_nl   = df_use['Zeit (s)'].values
            _sig_nl = df_use[active_sensor].fillna(0).values
            _mask_nl = (_t_nl >= min(xa, xb)) & (_t_nl <= max(xa, xb))
            if np.any(_mask_nl):
                _y_int_nl = np.cumsum(_sig_nl[_mask_nl]) * float(_t_nl[1] - _t_nl[0]) * _vis_scale
                _y_int_nl -= _y_int_nl[0]
                _alle_auto_bereiche['__int__'] = (float(_y_int_nl.min()), float(_y_int_nl.max()))

    _mc_sel_nl = st.session_state.get('multi_kanal_auswahl', [])
    if show_multi_diag and len(_mc_sel_nl) == 2:
        if float(st.session_state.get('multi_diag_ymin', 0)) == 0 and float(st.session_state.get('multi_diag_ymax', 0)) == 0:
            _mc_a_nl, _mc_b_nl = _mc_sel_nl
            if _mc_a_nl in df_use.columns and _mc_b_nl in df_use.columns and len(df_use) > 1:
                _t_nl    = df_use['Zeit (s)'].values
                _prod_nl = df_use[_mc_a_nl].values * df_use[_mc_b_nl].values
                _mask_nl = (_t_nl >= min(xa, xb)) & (_t_nl <= max(xa, xb))
                if np.any(_mask_nl):
                    _y_md_nl  = np.cumsum(_prod_nl[_mask_nl]) * float(_t_nl[1] - _t_nl[0]) * _vis_scale
                    _y_md_nl -= _y_md_nl[0]
                    _alle_auto_bereiche['__md__'] = (float(_y_md_nl.min()), float(_y_md_nl.max()))

    if len(_alle_auto_bereiche) > 1:
        # Maximales positives und negatives Verhältnis über alle Achsen bestimmen.
        _max_ratio_pos = 0.0
        _max_ratio_neg = 0.0
        for _dlo, _dhi in _alle_auto_bereiche.values():
            _dhi_pb = _dhi * (1 + Y_PUFFER)
            _dlo_pb = _dlo * (1 + Y_PUFFER)
            _pos    = max(_dhi_pb, 0.0)
            _neg    = max(-_dlo_pb, 0.0)
            _span   = _pos + _neg
            if not (_span > 0):   # fängt auch NaN ab (NaN > 0 ist False)
                continue
            _max_ratio_pos = max(_max_ratio_pos, _pos / _span)
            _max_ratio_neg = max(_max_ratio_neg, _neg / _span)

        _ratio_sum = _max_ratio_pos + _max_ratio_neg
        if _ratio_sum <= 0:
            pass
        elif _max_ratio_pos == 0 or _max_ratio_neg == 0:
            # Alle Achsen liegen nur auf einer Seite – normale Automatik reicht.
            pass
        else:
            _max_ratio_pos /= _ratio_sum
            _max_ratio_neg /= _ratio_sum

            for _n, (_dlo, _dhi) in _alle_auto_bereiche.items():
                _pos_raw   = max(_dhi * (1 + Y_PUFFER), 0.0)
                _neg_raw   = max(-_dlo * (1 + Y_PUFFER), 0.0)
                _scale_min = max(
                    _pos_raw / _max_ratio_pos if _max_ratio_pos > 0 else 0.0,
                    _neg_raw / _max_ratio_neg if _max_ratio_neg > 0 else 0.0,
                )
                _scale = _runde_scale(_scale_min)
                _nulllinie_ranges[_n] = (-_scale * _max_ratio_neg, _scale * _max_ratio_pos)

velocity_ok     = velocity is not None
acceleration_ok = acceleration is not None
_kanal_farbe_map = {name: KANAL_FARBEN[sensor_namen.index(name)] for name in sichtbare_sensor_namen}
_int_einheit = f'{_aktiv_einheit}·{_vis_einheit}'
_mc_auswahl_diag = st.session_state.get('multi_kanal_auswahl', [])
_mc_diag_eu = (
    _produkt_einheit(kanal_einheit_map.get(_mc_auswahl_diag[0],'?'),
                     kanal_einheit_map.get(_mc_auswahl_diag[1],'?')) + f'·{_vis_einheit}'
    if show_multi_diag and len(_mc_auswahl_diag) == 2 else ''
)
_y_limits: dict[str, float] = {
    **{f'ch{i}_ymin': float(st.session_state.get(f'ch{i}_ymin', 0)) for i in range(1, N_KANÄLE + 1)},
    **{f'ch{i}_ymax': float(st.session_state.get(f'ch{i}_ymax', 0)) for i in range(1, N_KANÄLE + 1)},
    'v_axis_min':      float(st.session_state.get('v_axis_min', 0)),
    'v_axis_max':      float(st.session_state.get('v_axis_max', 0)),
    'a_axis_min':      float(st.session_state.get('a_axis_min', 0)),
    'a_axis_max':      float(st.session_state.get('a_axis_max', 0)),
    'int_axis_min':    float(st.session_state.get('int_axis_min', 0)),
    'int_axis_max':    float(st.session_state.get('int_axis_max', 0)),
    'multi_diag_ymin': float(st.session_state.get('multi_diag_ymin', 0)),
    'multi_diag_ymax': float(st.session_state.get('multi_diag_ymax', 0)),
}
kanal_zu_yaxis, layout_yachsen, v_yaxis, a_yaxis, int_yaxis, multi_diag_yaxis, x_domain_end = _yachsen_layout(
    sichtbare_sensor_namen, kanal_einheit_map, y_range_plot,
    show_velocity, velocity_ok, show_acceleration, acceleration_ok,
    v_einheit=v_einheit, a_einheit=a_einheit,
    kanal_farbe_map=_kanal_farbe_map,
    y_ranges_fallback=_yrange_fallback,
    kanal_bereiche=_kanal_bereiche,
    kanal_ch_num=_sensor_ch_num,
    show_integral_curve=show_integral_curve,
    int_einheit=_int_einheit,
    show_multi_diag=show_multi_diag and len(_mc_auswahl_diag) == 2,
    multi_diag_einheit=_mc_diag_eu,
    nulllinie_ranges=_nulllinie_ranges if _nulllinie_ranges else None,
    y_limits=_y_limits,
)
active_yaxis = kanal_zu_yaxis.get(active_sensor, 'y')

# Y-Achsen-Skalenbereich des aktiven Kanals für Y-Slider speichern (nächster Renderdurchlauf)
_ys_layout_key = 'yaxis' if active_yaxis == 'y' else f'yaxis{active_yaxis[1:]}'
_ys_axis_range = layout_yachsen.get(_ys_layout_key, {}).get('range', None)
if _ys_axis_range and len(_ys_axis_range) == 2:
    st.session_state['_ys_axis_lo'] = float(_ys_axis_range[0])
    st.session_state['_ys_axis_hi'] = float(_ys_axis_range[1])

fig = go.Figure()

# Relative Zeitachse + automatische Einheitenskalierung:
# (t - t_offset) × _vis_scale → Anzeigewert in _vis_einheit.
# Daten bleiben intern in Sekunden (SI); Chart-Variablen _c sind nur für die Anzeige.
def _sc(t): return (t - t_offset) * _vis_scale if t is not None else None
df_plot_c = df_plot.copy()
df_plot_c['Zeit'] = (df_plot_c['Zeit (s)'] - t_offset) * _vis_scale
df_plot_c = df_plot_c.drop(columns=['Zeit (s)'])
xa_c, xb_c             = _sc(xa),            _sc(xb)
min_c, max_c            = _sc(min_zeit),      _sc(max_zeit)
sop_c = [(_sc(t), _sc(t0), _sc(t1), y) for t, t0, t1, y in sop_linien]
t_vmax_start_c, t_vmax_ende_c = _sc(t_vmax_start), _sc(t_vmax_ende)
t_vmin_start_c, t_vmin_ende_c = _sc(t_vmin_start), _sc(t_vmin_ende)
t_amax_f_c, t_amax_r_c        = _sc(t_amax_falling), _sc(t_amax_rising)

if rect_fit is not None:
    rect_fit_c = {
        **rect_fit,
        'runs': [
            {**run, 't_start': _sc(run['t_start']), 't_end': _sc(run['t_end'])}
            for run in rect_fit['runs']
        ],
    }
else:
    rect_fit_c = None

_baue_traces(
    fig, df_plot_c, sichtbare_sensor_namen,
    kanal_zu_yaxis, active_sensor, active_yaxis,
    sensor_namen,
    xa_c, xb_c, ya, yb, min_c, max_c,
    show_v_avg, rect_fit_c, show_rect_fit, show_rect_fit_top,
    has_vmax and mark_vmax, t_vmax_start_c, y_vmax_start, t_vmax_ende_c, y_vmax_ende,
    has_vmin and mark_vmin, t_vmin_start_c, y_vmin_start, t_vmin_ende_c, y_vmin_ende,
    has_amax_falling and mark_afall, t_amax_f_c, y_amax_falling,
    has_amax_rising  and mark_arise,  t_amax_r_c, y_amax_rising,
    sop_c,
    show_velocity, velocity, v_yaxis,
    show_acceleration, acceleration, a_yaxis,
    show_integral, int_yaxis, show_integral_curve,
)

_y_lim_keys = (
    ['v_axis_min', 'v_axis_max', 'a_axis_min', 'a_axis_max']
    + [f'ch{_i}_ymin' for _i in range(1, N_KANÄLE + 1)]
    + [f'ch{_i}_ymax' for _i in range(1, N_KANÄLE + 1)]
)
_y_lim_token = hash(tuple(st.session_state.get(k, 0) for k in _y_lim_keys))
_axis_token  = hash(tuple(sorted(kanal_zu_yaxis.items())))

_x_titel = f"Zeit rel. ({_vis_einheit})" if t_offset != 0.0 else f"Zeit ({_vis_einheit})"
fig.update_layout(
    xaxis_title=_x_titel,
    height=600,
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
    uirevision=f"{st.session_state.zoom_token}-{st.session_state.crop_start}-{st.session_state.crop_end}-{_y_lim_token}-{_axis_token}-{round(max_zeit_full, 4)}-{t_offset}-{_vis_scale}-{st.session_state.window_length}-{st.session_state.window_length_accel}-{v_einheit}-{a_einheit}-{active_sensor}",
    xaxis=dict(autorange=True,
               rangemode='normal' if t_offset != 0.0 else 'nonnegative',
               domain=[0, x_domain_end],
               showgrid=True, gridcolor='rgba(180,180,180,0.4)', gridwidth=1, nticks=20),
    **layout_yachsen,
)
# ∫IN1×IN2 dt Verlauf im Diagramm — kumulatives Integral des Produkts zwischen XA und XB
if show_multi_diag and len(_mc_auswahl_diag) == 2:
    _mc_a_d, _mc_b_d = _mc_auswahl_diag
    if _mc_a_d in df_plot_c.columns and _mc_b_d in df_plot_c.columns:
        _prod_t    = df_plot_c['Zeit'].values
        _prod_inst = df_plot_c[_mc_a_d].values * df_plot_c[_mc_b_d].values
        _dt_prod   = float(_prod_t[1] - _prod_t[0]) if len(_prod_t) > 1 else 1.0
        _mask_ab   = (_prod_t >= min(xa_c, xb_c)) & (_prod_t <= max(xa_c, xb_c))
        if np.any(_mask_ab):
            _x_ab = _prod_t[_mask_ab]
            _y_ab = np.cumsum(_prod_inst[_mask_ab]) * _dt_prod
            _y_ab -= _y_ab[0]             # Startwert 0 am linken Slider
            fig.add_trace(go.Scatter(
                x=_x_ab, y=_y_ab,
                mode='lines', name=f'∫{_mc_a_d}×{_mc_b_d} dt',
                line=dict(color='rgba(130,0,200,0.85)', width=2),
                yaxis=multi_diag_yaxis,
            ))
            _zeichne_integral_flaeche(fig, _x_ab, _y_ab, yaxis=multi_diag_yaxis)

if show_y_slider:
    _ya_line = float(st.session_state.get('ya_sw', 0.0))
    _yb_line = float(st.session_state.get('yb_sw', 0.0))
    fig.add_hline(y=_ya_line, line_dash="dash", line_color=FARBE_CURSOR,
                  annotation_text=f"YA {_ya_line:.2f}", annotation_position="top left")
    fig.add_hline(y=_yb_line, line_dash="dash", line_color="darkorange",
                  annotation_text=f"YB {_yb_line:.2f}", annotation_position="top left")

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
    _sl_min   = round((min_zeit - t_offset) * _vis_scale, 6)
    _sl_max   = round((max_zeit - t_offset) * _vis_scale, 6)
    _sl_range = abs(_sl_max - _sl_min)
    # Adaptiver Schritt: ca. 1000 Positionen über den sichtbaren Bereich.
    _sl_step  = max(1e-9, _sl_range / 1000.0) if _sl_range > 1e-9 else 1e-3
    _sl_step  = round(_sl_step, 9)
    _sl_fmt   = f"%.3f {_vis_einheit}"
    st.slider(
        "XA", _sl_min, _sl_max,
        key="xa_sw", step=_sl_step, format=_sl_fmt,
        on_change=update_xa_from_slider, label_visibility="collapsed",
        help=f"Linker Cursor XA ({_vis_einheit}) – ziehen oder Wert im Expander 'Diagrammarker' eingeben.",
    )
    st.slider(
        "XB", _sl_min, _sl_max,
        key="xb_sw", step=_sl_step, format=_sl_fmt,
        on_change=update_xb_from_slider, label_visibility="collapsed",
        help=f"Rechter Cursor XB ({_vis_einheit}) – ziehen oder Wert im Expander 'Diagrammarker' eingeben.",
    )

# ---------------------------------------------------------------------------
# CROP / SHOW ALL
# ---------------------------------------------------------------------------

margin  = abs(xb - xa) * CROP_RAND_FAKTOR
crop_t0 = max(min_zeit, min(xa, xb) - margin)
crop_t1 = min(max_zeit, max(xa, xb) + margin)

btn_col0, btn_col1, btn_col2 = st.columns([1, 3, 4])
with btn_col0:
    st.toggle(" ", key="x_rel_mode", on_change=_on_x_rel_mode_change,
              help="Start @ 0 – Relative Zeitachse: XA-Position beim Aktivieren wird als Nullpunkt eingefroren. Deaktivieren und neu aktivieren setzt den Nullpunkt auf die aktuelle XA-Position.")
with btn_col1:
    if st.button("✂️ Crop A–B  (+15%)", disabled=(dt_val_s == 0), width="stretch",
                 help="Schneidet die Ansicht auf den Bereich zwischen XA und XB zu (je 15 % Rand beiderseits)."):
        st.session_state.crop_start = crop_t0
        st.session_state.crop_end   = crop_t1
        _xa_new = round(float(min(xa, xb)), 9)
        _xb_new = round(float(max(xa, xb)), 9)
        st.session_state.xa = _xa_new
        st.session_state.xb = _xb_new
        st.session_state.zoom_token += 1
        st.rerun()
with btn_col2:
    if st.button("🔍 Alles zeigen | Show All", disabled=not crop_active, width="stretch",
                 help="Setzt den Crop zurück und zeigt den gesamten Messzeitraum."):
        st.session_state.crop_start  = None
        st.session_state.crop_end    = None
        st.session_state.zoom_token += 1
        st.rerun()

if crop_active:
    _cs_disp = st.session_state.crop_start - t_offset   # Sekunden
    _ce_disp = st.session_state.crop_end   - t_offset   # Sekunden
    st.caption(
        f"✂️ Crop aktiv: {_fmt_zeit(_cs_disp, 's')} – {_fmt_zeit(_ce_disp, 's')}"
    )

# ---------------------------------------------------------------------------
# KENNGRÖSSEN-ANZEIGE
# ---------------------------------------------------------------------------

freq_hz = (1.0 / dt_val_s) if dt_val_s > 0 else float('nan')
hub     = abs(rect_fit['y_high'] - rect_fit['y_low']) if rect_fit is not None else float('nan')

# Zeile 1 – Zeit & Signal
z1, z2, z3, z4 = st.columns(4)
z1.metric("Δt (A-B)",          _fmt_zeit(dt_val_s, 's'))
z2.metric("Frequenz Δt (A-B)", _fmt_freq(freq_hz))
z3.metric("Δs (A-B)",          _fmt_val(dy, _aktiv_einheit))
z4.metric("Hub Best-fit",      _fmt_val(hub, _aktiv_einheit) if not np.isnan(hub) else "N/A")

# Zeile 2 – 1. Ableitung (v_max/v_min wenn Linie aktiv oder Marker gesetzt)
g1, g2, g3, g4 = st.columns(4)
g1.metric(f"d{active_sensor}/dt (A-B)",   _fmt_val(v_avg, v_einheit))
g2.metric(f"Δd{active_sensor}/dt (A-B)",  _fmt_val(v_cursor_delta, v_einheit) if not np.isnan(v_cursor_delta) else "N/A")
if show_velocity or (mark_vmax and has_vmax):
    g3.metric(f"d{active_sensor}/dt max", _fmt_val(v_max, v_einheit) if not np.isnan(v_max) else "N/A")
if show_velocity or (mark_vmin and has_vmin):
    g4.metric(f"d{active_sensor}/dt min", _fmt_val(v_min, v_einheit) if not np.isnan(v_min) else "N/A")

# Zeile 3 – 2. Ableitung + Integral + Multi-Kanal-Integral (a-Werte wenn Linie aktiv oder Marker gesetzt)
_mc_auswahl = st.session_state.get('multi_kanal_auswahl', [])
a1, a2, a3, a4 = st.columns(4)
if show_acceleration or (mark_afall and has_amax_falling):
    a1.metric(f"d²{active_sensor}/dt² max Fall.", _fmt_val(a_max_falling, a_einheit) if not np.isnan(a_max_falling) else "N/A")
if show_acceleration or (mark_arise and has_amax_rising):
    a2.metric(f"d²{active_sensor}/dt² min Rise.", _fmt_val(a_min_rising, a_einheit)  if not np.isnan(a_min_rising) else "N/A")
if show_integral:
    a3.metric(f"∫{active_sensor} dt (A-B)",   _fmt_integral(integral_val, _aktiv_einheit, 's'))
if show_multi_kanal and len(_mc_auswahl) == 2:
    _mc_einh_a = kanal_einheit_map.get(_mc_auswahl[0], '?')
    _mc_einh_b = kanal_einheit_map.get(_mc_auswahl[1], '?')
    a4.metric(f"∫{_mc_auswahl[0]}×{_mc_auswahl[1]} dt (A-B)",
              _fmt_integral(multi_integral_val, _produkt_einheit(_mc_einh_a, _mc_einh_b), 's'))

# Zeile 4 – SOP (nur wenn aktiviert und Wert vorhanden)
if show_sop and not np.isnan(v_sop):
    s1, *_ = st.columns(4)
    s1.metric(f"SOP d{active_sensor}/dt ({st.session_state.get('sop_percent', 80)} %)", _fmt_val(v_sop, v_einheit))

# Zeile 5 – Y-Slider Delta (nur wenn aktiviert)
if show_y_slider:
    _ya_m = float(st.session_state.get('ya_sw', 0.0))
    _yb_m = float(st.session_state.get('yb_sw', 0.0))
    _y_delta = abs(_yb_m - _ya_m)
    yd1, yd2, yd3, *_ = st.columns(4)
    yd1.metric("YA", f"{_ya_m:.2f} {_aktiv_einheit}")
    yd2.metric("YB", f"{_yb_m:.2f} {_aktiv_einheit}")
    yd3.metric("ΔYA–YB", f"{_y_delta:.2f} {_aktiv_einheit}")

# Zeile 6 – Widerstands-Integral (nur wenn aktiviert)
if show_widerstand_integral and _w_kanal:
    _w_kohm_d = float(st.session_state.get('widerstand_kohm', 200.0))
    w1, *_ = st.columns(4)
    w1.metric(
        f"∫{_w_kanal}/R dt (A-B)",
        _fmt_integral(widerstand_integral_val, 'A', 's'),
        help=f"I = U_{_w_kanal} / {_w_kohm_d:.3g} kΩ",
    )

# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------

st.sidebar.header("3. Export")
metrics = {
    # Zeit & Signal
    "XA":                       _fmt_zeit(xa - t_offset, 's'),
    "XB":                       _fmt_zeit(xb - t_offset, 's'),
    "Δt (A-B)":                 _fmt_zeit(dt_val_s, 's'),
    "Frequenz Δt (A-B)":        _fmt_freq(freq_hz),
    "Δs (A-B)":                 _fmt_val(dy, _aktiv_einheit),
    "Hub Best-fit":             _fmt_val(hub, _aktiv_einheit)              if not np.isnan(hub) else "N/A",
    # 1. Ableitung (Cursor-basierte Werte immer vorhanden)
    f"d {active_sensor} /dt (A-B)":   _fmt_val(v_avg, v_einheit),
    f"Δd {active_sensor} /dt (A-B)":  _fmt_val(v_cursor_delta, v_einheit)   if not np.isnan(v_cursor_delta) else "N/A",
}
if show_integral:
    metrics[f"∫ {active_sensor} dt (A-B)"] = _fmt_integral(integral_val, _aktiv_einheit, 's')
# SG-Peak-Werte wenn Marker aktiv (unabhängig von d/dt-Linie)
if has_vmax and mark_vmax:
    metrics[f"d {active_sensor} /dt max"] = _fmt_val(v_max, v_einheit) if not np.isnan(v_max) else "N/A"
if has_vmin and mark_vmin:
    metrics[f"d {active_sensor} /dt min"] = _fmt_val(v_min, v_einheit) if not np.isnan(v_min) else "N/A"
if has_amax_falling and mark_afall:
    metrics[f"d² {active_sensor} /dt² max Fall."] = _fmt_val(a_max_falling, a_einheit) if not np.isnan(a_max_falling) else "N/A"
if has_amax_rising and mark_arise:
    metrics[f"d² {active_sensor} /dt² min Rise."] = _fmt_val(a_min_rising, a_einheit)  if not np.isnan(a_min_rising) else "N/A"
# Multi-Kanal-Integral in Export aufnehmen (nur wenn Ergebnis vorhanden)
if show_multi_kanal and len(st.session_state.get('multi_kanal_auswahl', [])) == 2:
    _mc_a_exp   = st.session_state['multi_kanal_auswahl'][0]
    _mc_b_exp   = st.session_state['multi_kanal_auswahl'][1]
    _mc_ea_exp  = kanal_einheit_map.get(_mc_a_exp, '?')
    _mc_eb_exp  = kanal_einheit_map.get(_mc_b_exp, '?')
    metrics[f"∫ {_mc_a_exp}×{_mc_b_exp} dt (A-B)"] = _fmt_integral(
        multi_integral_val, _produkt_einheit(_mc_ea_exp, _mc_eb_exp), 's'
    )
if show_widerstand_integral and _w_kanal and not np.isnan(widerstand_integral_val):
    _w_kohm_exp = float(st.session_state.get('widerstand_kohm', 200.0))
    metrics[f"∫{_w_kanal}/R dt (A-B) [R={_w_kohm_exp:.3g} kΩ]"] = _fmt_integral(
        widerstand_integral_val, 'A', 's'
    )
if show_sop and not np.isnan(v_sop):
    metrics[f"SOP d{active_sensor}/dt ({st.session_state.get('sop_percent', 80)} %)"] = _fmt_val(v_sop, v_einheit)
if show_y_slider:
    _ya_exp = float(st.session_state.get('ya_sw', 0.0))
    _yb_exp = float(st.session_state.get('yb_sw', 0.0))
    metrics["YA"] = f"{_ya_exp:.2f} {_aktiv_einheit}"
    metrics["YB"] = f"{_yb_exp:.2f} {_aktiv_einheit}"
    metrics["ΔYA–YB"] = f"{abs(_yb_exp - _ya_exp):.2f} {_aktiv_einheit}"
_multi_diag_export = None
if show_multi_diag and len(_mc_auswahl_diag) == 2:
    _mc_d0, _mc_d1 = _mc_auswahl_diag
    if _mc_d0 in df.columns and _mc_d1 in df.columns:
        _t_md_all  = (df['Zeit (s)'].values - t_offset) * _vis_scale   # Anzeigeeinheit
        _y_md_all  = df[_mc_d0].values * df[_mc_d1].values
        # Auf XA–XB maskieren – identisch mit der Bildschirmansicht (app.py _mask_ab, Z~1858)
        _mask_md   = (_t_md_all >= min(xa_c, xb_c)) & (_t_md_all <= max(xa_c, xb_c))
        _t_md_exp  = _t_md_all[_mask_md] if np.any(_mask_md) else _t_md_all
        _y_md_exp  = _y_md_all[_mask_md] if np.any(_mask_md) else _y_md_all
        _multi_diag_export = {
            'x':      _t_md_exp,
            'y':      _y_md_exp,
            'einheit': _produkt_einheit(kanal_einheit_map.get(_mc_d0,'?'), kanal_einheit_map.get(_mc_d1,'?')),
            'label':  f"{_mc_d0}×{_mc_d1}",
            'ymin':   float(st.session_state.get('multi_diag_ymin', 0)),
            'ymax':   float(st.session_state.get('multi_diag_ymax', 0)),
        }

# --- Export-Format-Auswahl ---
st.sidebar.radio(
    "Format", ["PDF", "PNG", "CSV (Oszi)", "CSV (plain)", "CSX"], horizontal=True,
    key="export_format", label_visibility="collapsed",
)
_exp_format = st.session_state.get("export_format", "PDF")
_exp_ist_diagramm = _exp_format in ("PDF", "PNG")

_exp_mit_werten = st.sidebar.toggle(
    "Mit Werten", key="export_mit_werten",
    help="Messwerte in PDF-Tabelle / PNG-Annotation einbetten.",
    disabled=not _exp_ist_diagramm,
)

if _exp_mit_werten and _exp_ist_diagramm:
    _alle_metrik_keys = list(metrics.keys())
    _gespeicherte_auswahl = st.session_state.get('export_wert_auswahl')

    # _bekannte_keys: Keys die beim letzten Render vorhanden waren.
    # Beim ersten Render (Session-Start oder nach Settings-Load) auf alle aktuellen
    # Keys initialisieren, damit nur echte Neuzugänge auto-ergänzt werden und
    # die gespeicherte Auswahl (inkl. Abwahlen) erhalten bleibt.
    _bk_raw = st.session_state.get('export_wert_bekannte_keys')
    _bekannte_keys = set(_bk_raw) if _bk_raw is not None else set(_alle_metrik_keys)

    if _gespeicherte_auswahl is None:
        _vorauswahl = _alle_metrik_keys
    else:
        _saved_valid = [k for k in _gespeicherte_auswahl if k in set(_alle_metrik_keys)]
        # Nur Keys, die seit dem letzten Render erstmals aufgetaucht sind (z. B. frisch
        # aktivierter Marker), werden automatisch ergänzt. Explizit abgewählte Keys
        # bleiben abgewählt, weil sie in _bekannte_keys stehen.
        _neue_keys   = [k for k in _alle_metrik_keys if k not in _bekannte_keys]
        _vorauswahl  = _saved_valid + _neue_keys

    # Bekannte Keys für nächsten Render festhalten (vor Widget-Rendering setzen)
    st.session_state['export_wert_bekannte_keys'] = _alle_metrik_keys
    st.session_state['export_wert_auswahl'] = _vorauswahl
    st.sidebar.multiselect(
        "Exportierte Werte", _alle_metrik_keys,
        key="export_wert_auswahl",
        help="Welche Messwerte sollen exportiert werden? Die Auswahl wird mit den Einstellungen gespeichert/geladen.",
    )
with st.sidebar.expander("Beschreibung", expanded=False):
    st.text_input(
        "Überschrift",
        key="pdf_titel",
        max_chars=50,
        help="Titelzeile im PDF (max. 50 Zeichen). Leer = automatisch aus Dateiname.",
    )
    st.text_area(
        "Beschreibung",
        key="pdf_beschreibung",
        max_chars=200,
        height=80,
        help="Wird in kleinerer Schrift unter der Überschrift im PDF angezeigt (max. 200 Zeichen).",
    )

if st.sidebar.button("📥 Export erstellen", width="stretch",
                     help="Erstellt die Exportdatei im gewählten Format – Download-Button erscheint danach."):
    with st.spinner("Wird erstellt..."):
        try:
            stem = uploaded_file.name.rsplit('.', 1)[0]
            _ex_kanäle  = sichtbare_sensor_namen
            _valid_mask = df[_ex_kanäle].notna().all(axis=1)
            _df_ex      = df[_valid_mask].reset_index(drop=True)

            if _exp_format == "CSV (Oszi)":
                _n_ex      = len(_ex_kanäle)
                _zeit_s    = _df_ex['Zeit (s)'].values - t_offset
                _data_mat  = np.column_stack([_zeit_s] + [_df_ex[k].values for k in _ex_kanäle])
                _buf       = io.StringIO()
                np.savetxt(_buf, _data_mat, fmt=['%.10e'] + ['%.8e'] * _n_ex, delimiter=',')
                _csv_bytes = (
                    'x-axis,' + ','.join(str(i + 1) for i in range(_n_ex)) + '\n'
                    + 'second,' + ','.join(kanal_einheit_map.get(k, '') for k in _ex_kanäle) + '\n'
                    + _buf.getvalue()
                ).encode('utf-8')
                st.sidebar.download_button("💾 CSV herunterladen", _csv_bytes,
                    file_name=f"{stem}_osc.csv", mime="text/csv", width="stretch")

            elif _exp_format == "CSV (plain)":
                _csv_bytes = (
                    ','.join(_ex_kanäle) + '\n'
                    + ','.join(kanal_einheit_map.get(k, '') for k in _ex_kanäle) + '\n'
                    + _df_ex[_ex_kanäle].to_csv(index=False, header=False, float_format='%.8e')
                ).encode('utf-8')
                st.sidebar.download_button("💾 CSV herunterladen", _csv_bytes,
                    file_name=f"{stem}_plain.csv", mime="text/csv", width="stretch")

            elif _exp_format == "CSX":
                _csx_cfg = {k: st.session_state[k] for k in EINSTELLUNGEN_KEYS if k in st.session_state}
                _csx_json = json.dumps(_csx_cfg, ensure_ascii=False).encode('utf-8')
                _csx_bytes = (
                    file_bytes
                    + b"\n###CSX_SETTINGS_BEGIN###\n"
                    + _csx_json
                    + b"\n###CSX_SETTINGS_END###\n"
                )
                st.sidebar.download_button("💾 CSX herunterladen", _csx_bytes,
                    file_name=f"{stem}.csx", mime="text/plain", width="stretch")

            else:
                # --- Diagramm-Export (PDF / PNG) ---
                _wert_auswahl_raw = st.session_state.get('export_wert_auswahl')
                _wert_auswahl = _wert_auswahl_raw if _wert_auswahl_raw is not None else list(metrics.keys())
                metrics_export = (
                    {k: v for k, v in metrics.items() if k in _wert_auswahl}
                    if _exp_mit_werten and _wert_auswahl else {}
                )
                _df_exp_c = df.copy()
                _df_exp_c['Zeit'] = (df['Zeit (s)'] - t_offset) * _vis_scale
                _df_exp_c = _df_exp_c.drop(columns=['Zeit (s)'])
                chart_png = build_chart_png(
                    _df_exp_c, sichtbare_sensor_namen, active_sensor,
                    xa_c, xb_c, ya, yb, show_v_avg,
                    t_vmax_start_c, y_vmax_start, t_vmax_ende_c, y_vmax_ende, has_vmax and mark_vmax,
                    t_vmin_start_c, y_vmin_start, t_vmin_ende_c, y_vmin_ende, has_vmin and mark_vmin,
                    t_amax_f_c, y_amax_falling, has_amax_falling and mark_afall,
                    t_amax_r_c, y_amax_rising,  has_amax_rising  and mark_arise,
                    show_rect_fit=show_rect_fit,
                    show_rect_fit_top=show_rect_fit_top,
                    rect_fit=rect_fit,
                    show_velocity=show_velocity,
                    window_length=st.session_state.window_length,
                    velocity_data=(sg_v_roh * v_faktor) if show_velocity and sg_v_roh is not None else None,
                    show_acceleration=show_acceleration,
                    window_length_accel=st.session_state.window_length_accel,
                    acceleration_data=(sg_a_roh * a_faktor) if show_acceleration and sg_a_roh is not None else None,
                    v_einheit_label=v_einheit if show_velocity else None,
                    a_einheit_label=a_einheit if show_acceleration else None,
                    sop_linien=sop_c,
                    kanal_einheit_map=kanal_einheit_map,
                    alle_sensor_namen=sensor_namen,
                    hz_faktor=_vis_hz,
                    zeit_einheit=_vis_einheit,
                    kanal_ch_num=_sensor_ch_num,
                    show_integral=show_integral,
                    show_integral_curve=show_integral_curve,
                    multi_diag_data=_multi_diag_export,
                    show_y_slider=show_y_slider,
                    y_slider_a=float(st.session_state.get('ya_sw', 0.0)),
                    y_slider_b=float(st.session_state.get('yb_sw', 0.0)),
                    kanal_bereiche=_kanal_bereiche,
                    y_ranges_fallback=_yrange_fallback,
                    metrics=metrics_export if _exp_format == "PNG" else {},
                    nulllinie_ranges=_nulllinie_ranges if _nulllinie_ranges else None,
                    y_limits=_y_limits,
                    x_rangemode='normal' if t_offset != 0.0 else 'nonnegative',
                )
                if _exp_format == "PDF":
                    file_bytes_out = build_pdf(
                        uploaded_file.name, chart_png, metrics_export,
                        pdf_titel=st.session_state.get('pdf_titel', ''),
                        pdf_beschreibung=st.session_state.get('pdf_beschreibung', ''),
                    )
                    st.sidebar.download_button("💾 PDF herunterladen", file_bytes_out,
                        file_name=f"{stem}_auswertung.pdf", mime="application/pdf", width="stretch")
                else:
                    st.sidebar.download_button("💾 PNG herunterladen", chart_png,
                        file_name=f"{stem}_diagramm.png", mime="image/png", width="stretch")

        except Exception as exc:
            st.sidebar.error(f"Export fehlgeschlagen: {exc}")
