"""Diagramm- und Export-Funktionen für die Messdaten-Auswertung.

Enthält Plotly-Chart-Aufbau, Kaleido-PNG-Export, PDF-Export
und alle zugehörigen Berechnungshilfen (keine Streamlit-UI-Widgets).
"""
import io
import datetime

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

import reader

# ---------------------------------------------------------------------------
# KONSTANTEN
# ---------------------------------------------------------------------------

SPLIT_FAKTOR = 15.0  # Y-Achsen-Aufteilung bei gleicher Einheit wenn Bereiche > Faktor abweichen

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
FARBE_CURSOR        = 'red'
FARBE_RECHTECK      = 'lime'
FARBE_INTEGRAL_POS  = 'rgba(0, 100, 200, 0.20)'   # blau – Bereich über der 0-Linie
FARBE_INTEGRAL_NEG  = 'rgba(200, 50, 0, 0.20)'    # rot  – Bereich unter der 0-Linie

_ZEIT_TO_S: dict[str, float] = {'s': 1.0, 'ms': 1e-3, 'µs': 1e-6, 'ns': 1e-9}


# ---------------------------------------------------------------------------
# ABLEITUNGS-EINHEITEN
# ---------------------------------------------------------------------------

def _ableit_info(einheit: str, zeit_einheit: str = 'ms') -> tuple[str, str, float, float]:
    """(v_einheit, a_einheit, v_faktor, a_faktor) für eine Kanal-Einheit.

    v_faktor/a_faktor wandeln SG-Rohableitung (einheit/s, einheit/s²) in
    einheit/zeit_einheit bzw. einheit/zeit_einheit² um.
    """
    zhf = 1.0 / _ZEIT_TO_S.get(zeit_einheit, 1e-3)   # s⁻¹ → display_unit⁻¹
    return (f'{einheit}/{zeit_einheit}', f'{einheit}/{zeit_einheit}²',
            1.0 / zhf, 1.0 / (zhf ** 2))


# ---------------------------------------------------------------------------
# MULTI-ACHSEN-LAYOUT
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
    kanal_ch_num: dict[str, int] | None = None,
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
    _ch_num = kanal_ch_num or {}

    def _user_lim_kanal(name: str) -> tuple[float, float] | None:
        ch_i = _ch_num.get(name, 0)
        if ch_i == 0:
            return None
        lo = float(st.session_state.get(f'ch{ch_i}_ymin', 0))
        hi = float(st.session_state.get(f'ch{ch_i}_ymax', 0))
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
        for n in kanäle:
            lim = _user_lim_kanal(n)
            if lim is not None:
                return list(lim)
        return _fallback_rng(titel, kanäle)

    # --- Schritt 1: Einheiten-basierte Gruppen (Reihenfolge aus kanal_namen) ---
    einheit_gruppen: dict[str, list[str]] = {}
    for n in kanal_namen:
        einheit_gruppen.setdefault(kanal_einheit_map.get(n, 'µm'), []).append(n)

    # --- Schritt 2: Innerhalb jeder Einheitsgruppe nach Grenzwert aufteilen ---
    # Kanäle mit gleichem Grenzwert → gemeinsame Gruppe
    # Kanäle ohne Grenzwert → eigene Untergruppe (SPLIT_FAKTOR-Logik)
    # Kanäle mit unterschiedlichem Grenzwert → je eigene Gruppe
    pre_achsen: list[tuple[str, list[str]]] = []   # (einheit, kanal_liste)
    for einheit, kanäle in einheit_gruppen.items():
        lim_gruppen: dict[tuple[float, float], list[str]] = {}
        kein_lim: list[str] = []
        for n in kanäle:
            lim = _user_lim_kanal(n)
            if lim is not None:
                lim_gruppen.setdefault(lim, []).append(n)
            else:
                kein_lim.append(n)
        # Jede Grenzgruppe bekommt eine eigene Achse
        for gruppe in lim_gruppen.values():
            pre_achsen.append((einheit, gruppe))
        # Kanäle ohne Grenzen: SPLIT_FAKTOR anwenden
        if len(kein_lim) == 1:
            pre_achsen.append((einheit, kein_lim))
        elif len(kein_lim) > 1:
            spans = [s for n in kein_lim if (s := _kanal_span(n)) is not None and s > 0]
            if len(spans) >= 2 and max(spans) / min(spans) > SPLIT_FAKTOR:
                for n in kein_lim:
                    pre_achsen.append((einheit, [n]))
            else:
                pre_achsen.append((einheit, list(kein_lim)))

    # --- Schritt 3: Einheitenübergreifend zusammenführen wenn Grenzen identisch ---
    final_achsen: list[tuple[str, list[str]]] = []
    lim_zu_idx: dict[tuple[float, float], int] = {}

    for einheit, kanäle in pre_achsen:
        # Gruppe ist nur dann zusammenführbar wenn ALLE Kanäle dieselbe Grenze teilen
        lim_0 = _user_lim_kanal(kanäle[0]) if kanäle else None
        lim = lim_0 if (lim_0 is not None
                        and all(_user_lim_kanal(n) == lim_0 for n in kanäle[1:])) else None
        if lim is not None and lim in lim_zu_idx:
            idx = lim_zu_idx[lim]
            ex_titel, ex_kanäle = final_achsen[idx]
            ex_einheiten = [e.strip() for e in ex_titel.split(' / ')]
            if einheit not in ex_einheiten:
                ex_titel = ex_titel + ' / ' + einheit
            final_achsen[idx] = (ex_titel, ex_kanäle + kanäle)
        else:
            if lim is not None:
                lim_zu_idx[lim] = len(final_achsen)
            final_achsen.append((einheit, list(kanäle)))

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
    x_domain_end = max(0.5, 1.0 - STEP * n_right) if n_right >= 1 else 1.0

    # Primäre linke Achse
    titel0, kanäle0 = final_achsen[0] if final_achsen else ('µm', [])
    rng0 = _rng(titel0, kanäle0) if final_achsen else None
    layout_yachsen: dict = {
        'yaxis': dict(title=titel0, range=rng0 if rng0 else y_range_primaer,
                      **_achsfarbe(kanäle0))
    }

    for idx, (yk, title, rng, farb_dict) in enumerate(rechte_achsen):
        pos = x_domain_end + STEP * idx
        ax: dict = dict(title=title, overlaying='y', side='right', showgrid=False,
                        position=pos, anchor='free', **farb_dict)
        if rng:
            ax['range'] = rng
        layout_yachsen[yk] = ax

    return kanal_zu_yaxis, layout_yachsen, v_yaxis, a_yaxis, x_domain_end


# ---------------------------------------------------------------------------
# RECHTECK-FIT ZEICHNEN
# ---------------------------------------------------------------------------

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
# SPEED ON POINT
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
# INTEGRAL-FLÄCHE
# ---------------------------------------------------------------------------

def _zeichne_integral_flaeche(fig, x_vals, y_vals, yaxis: str = 'y') -> None:
    """Zeichnet Integralfläche mit zwei Farben: blau über 0, rot unter 0.

    Nulldurchgänge werden linear interpoliert, damit die Farbbereiche sauber trennen.
    """
    import numpy as np
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    if len(x) < 2:
        return

    # Nulldurchgänge interpolieren und in x/y einfügen
    x_list, y_list = list(x), list(y)
    inserts = []
    for i in range(len(y) - 1):
        if y[i] != y[i + 1] and (y[i] > 0) != (y[i + 1] > 0):
            t = y[i] / (y[i] - y[i + 1])
            inserts.append((i + 1, float(x[i] + t * (x[i + 1] - x[i]))))
    for offset, (pos, x_c) in enumerate(inserts):
        x_list.insert(pos + offset, x_c)
        y_list.insert(pos + offset, 0.0)

    xa = np.array(x_list)
    ya = np.array(y_list)

    fig.add_trace(go.Scatter(
        x=xa, y=np.maximum(ya, 0.0),
        fill='tozeroy', fillcolor=FARBE_INTEGRAL_POS,
        line=dict(color='rgba(0,100,200,0.45)', width=1),
        name='∫', yaxis=yaxis,
    ))
    fig.add_trace(go.Scatter(
        x=xa, y=np.minimum(ya, 0.0),
        fill='tozeroy', fillcolor=FARBE_INTEGRAL_NEG,
        line=dict(color='rgba(200,50,0,0.45)', width=1),
        name='∫⁻', yaxis=yaxis, showlegend=False,
    ))


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
    kanal_ch_num: dict | None = None,
    show_integral: bool = False,
) -> bytes:
    """Rendert das Diagramm mit Kaleido zu PNG-Bytes für den Export."""
    if kanal_einheit_map is None:
        kanal_einheit_map = {n: 'µm' for n in sensor_namen}

    _aktiv_e_e = kanal_einheit_map.get(active_sensor, 'µm')
    v_einheit_e, a_einheit_e, v_faktor_e, a_faktor_e = _ableit_info(_aktiv_e_e, zeit_einheit)

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
        kanal_ch_num=kanal_ch_num,
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
    if show_integral:
        _mask_e = (df['Zeit (ms)'] >= xa) & (df['Zeit (ms)'] <= xb)
        _df_int_e = df[_mask_e]
        if len(_df_int_e) > 0:
            _zeichne_integral_flaeche(
                export_fig,
                _df_int_e['Zeit (ms)'].values, _df_int_e[active_sensor].values,
                yaxis=active_yaxis_e,
            )

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
