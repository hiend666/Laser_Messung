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

SPLIT_FAKTOR  = 15.0   # Y-Achsen-Aufteilung bei gleicher Einheit wenn Bereiche > Faktor abweichen
STEP          = 0.07   # Plotly-Abstand zwischen rechten Y-Achsen (Anteil der Figure-Breite)
Y_PUFFER      = 0.15   # Y-Bereich-Puffer oben (15 %) für alle Achsenbereiche
X_DOMAIN_MIN  = 0.5    # Mindestbreite des Plot-Bereichs wenn viele rechte Achsen vorhanden

# Diagramm-Farben – Kanäle
FARBE_KANAL1    = '#003366'
FARBE_KANAL2    = '#4c78a8'
FARBE_KANAL3    = '#d62728'
FARBE_KANAL4    = '#2ca02c'
KANAL_FARBEN    = [FARBE_KANAL1, FARBE_KANAL2, FARBE_KANAL3, FARBE_KANAL4]

# Diagramm-Farben – Auswertung
FARBE_D             = 'purple'
FARBE_D2            = 'orange'
FARBE_V_SCHNITT     = 'green'
FARBE_VMAX          = 'red'
FARBE_VMIN          = 'royalblue'
FARBE_AMAX          = 'orange'
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
            return [lo, hi + span * Y_PUFFER]
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
        for gruppe in lim_gruppen.values():
            pre_achsen.append((einheit, gruppe))
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
    x_domain_end = max(X_DOMAIN_MIN, 1.0 - STEP * n_right) if n_right >= 1 else 1.0

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
    mit_top_linie: bool = True,
    yaxis: str = 'y',
):
    """Fügt Rechteck-Fit-Traces und optionale Füllformen zum Diagramm hinzu."""
    for idx, run in enumerate(rect_fit['runs']):
        clipped_start = max(run['t_start'], bereich_min)
        clipped_end   = min(run['t_end'],   bereich_max)
        if clipped_start >= clipped_end:
            continue
        if mit_top_linie:
            fig.add_trace(go.Scatter(
                x=[clipped_start, clipped_end],
                y=[rect_fit['y_high'], rect_fit['y_high']],
                mode='lines',
                name='Rechteck-Fit' if idx == 0 else None,
                showlegend=(idx == 0),
                line=dict(color=FARBE_RECHTECK, dash='dash', width=2),
                yaxis=yaxis,
            ))
            fig.add_trace(go.Scatter(
                x=[clipped_start, clipped_end],
                y=[rect_fit['y_low'], rect_fit['y_low']],
                mode='lines',
                name=None, showlegend=False,
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
    """Findet SOP-Punkte an steigenden UND fallenden Flanken des Rechteck-Fits.

    Für jeden Puls werden steigende (Pulsanfang) und fallende (Pulsende) Flanken
    gesucht. Steigende Flanken werden bevorzugt: der v_sop-Messwert kommt vom
    ersten steigenden Kreuzungspunkt; ist keiner vorhanden, vom ersten fallenden.

    Gibt (sop_linien, v_sop) zurück:
    - sop_linien: Liste von (t_sop, t_links, t_rechts, y_level)
    - v_sop:      D am bevorzugten Kreuzungspunkt (in Anzeigeeinheit), oder nan
    """
    hub = rect_fit['y_high'] - rect_fit['y_low']
    if hub <= 0:
        return [], float('nan')

    sop_level  = rect_fit['y_low'] + (sop_percent / 100.0) * hub
    n          = len(signal)
    ergebnisse = []   # (t_sop, t0, t1, y, v, flanke)  flanke: 'rise'|'fall'

    def _kreuzung(idx_abs: int, flanke: str):
        i0   = max(0, idx_abs - halbes_zeitfenster)
        i1   = min(n - 1, idx_abs + halbes_zeitfenster)
        dt_s = (i1 - i0) / sample_rate
        v    = ((signal[i1] - signal[i0]) * v_faktor) / dt_s if dt_s > 0 else float('nan')
        t    = float(zeit[idx_abs])
        t0   = float(zeit[max(0, idx_abs - 10)])
        t1   = float(zeit[min(n - 1, idx_abs + 10)])
        return (t, t0, t1, sop_level, v, flanke)

    for run in rect_fit['runs']:
        puls_dauer = max(0.1, run['t_end'] - run['t_start'])

        # Steigende Flanke: kurz vor Pulsstart bis erstes Drittel des Pulses
        idx_r = np.where((zeit >= run['t_start'] - 0.5) &
                         (zeit <= run['t_start'] + puls_dauer * 0.3))[0]
        if len(idx_r) >= 2:
            s = signal[idx_r]
            kpos = np.where((s[:-1] < sop_level) & (s[1:] >= sop_level))[0]
            if len(kpos):
                ergebnisse.append(_kreuzung(int(idx_r[kpos[0] + 1]), 'rise'))

        # Fallende Flanke: letztes Drittel des Pulses bis kurz nach Pulsende
        idx_f = np.where((zeit >= run['t_end'] - puls_dauer * 0.3) &
                         (zeit <= run['t_end'] + 0.5))[0]
        if len(idx_f) >= 2:
            s = signal[idx_f]
            kpos = np.where((s[:-1] >= sop_level) & (s[1:] < sop_level))[0]
            if len(kpos):
                ergebnisse.append(_kreuzung(int(idx_f[kpos[0] + 1]), 'fall'))

    if not ergebnisse:
        return [], float('nan')

    sop_linien = [(e[0], e[1], e[2], e[3]) for e in ergebnisse]
    # Messwert: erster steigender Punkt bevorzugt, sonst erster gefundener
    rise = [e for e in ergebnisse if e[5] == 'rise']
    v_sop_wert = rise[0][4] if rise else ergebnisse[0][4]
    return sop_linien, v_sop_wert


# ---------------------------------------------------------------------------
# INTEGRAL-FLÄCHE
# ---------------------------------------------------------------------------

def _zeichne_integral_flaeche(fig: go.Figure, x_vals, y_vals, yaxis: str = 'y') -> None:
    """Zeichnet Integralfläche mit zwei Farben: blau über 0, rot unter 0.

    Nulldurchgänge werden linear interpoliert, damit die Farbbereiche sauber trennen.
    """
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    if len(x) < 2:
        return

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
# TRACES – gemeinsame Aufbau-Funktion für interaktives Diagramm und PNG-Export
# ---------------------------------------------------------------------------

def _baue_traces(
    fig: go.Figure,
    df_plot,
    sensor_namen: list[str],
    kanal_zu_yaxis: dict[str, str],
    active_sensor: str,
    active_yaxis: str,
    alle_sensor_namen: list[str],
    xa: float, xb: float,
    ya: float, yb: float,
    min_zeit: float, max_zeit: float,
    show_v_avg: bool,
    rect_fit,
    show_rect_fit: bool,
    show_rect_fit_top: bool,
    has_vmax: bool,
    t_vmax_start, y_vmax_start, t_vmax_ende, y_vmax_ende,
    has_vmin: bool,
    t_vmin_start, y_vmin_start, t_vmin_ende, y_vmin_ende,
    has_amax_falling: bool,
    t_amax_falling, y_amax_falling,
    has_amax_rising: bool,
    t_amax_rising, y_amax_rising,
    sop_linien: list,
    show_velocity: bool, velocity,
    v_yaxis: str,
    show_acceleration: bool, acceleration,
    a_yaxis: str,
    show_integral: bool,
) -> None:
    """Fügt alle Mess- und Auswertungs-Traces zu fig hinzu.

    Wird sowohl vom interaktiven Diagramm (app.py) als auch vom PNG-Export
    (build_chart_png) aufgerufen – identische Darstellung in beiden Fällen.
    """
    for name in sensor_namen:
        _ci = alle_sensor_namen.index(name) if name in alle_sensor_namen else 0
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
                              mit_fuellung=show_rect_fit,
                              mit_top_linie=show_rect_fit_top,
                              yaxis=active_yaxis)

    if has_vmax:
        fig.add_trace(go.Scatter(
            x=[t_vmax_start, t_vmax_ende], y=[y_vmax_start, y_vmax_ende],
            mode='lines+markers', name='D-max',
            line=dict(color=FARBE_VMAX, width=4),
            marker=dict(color=FARBE_VMAX, size=14, symbol='circle',
                        line=dict(color=FARBE_VMAX, width=2)),
            yaxis=active_yaxis,
        ))
    if has_vmin:
        fig.add_trace(go.Scatter(
            x=[t_vmin_start, t_vmin_ende], y=[y_vmin_start, y_vmin_ende],
            mode='lines+markers', name='D-min',
            line=dict(color=FARBE_VMIN, width=4),
            marker=dict(color=FARBE_VMIN, size=14, symbol='circle',
                        line=dict(color=FARBE_VMIN, width=2)),
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

    if show_integral:
        _mask = (df_plot['Zeit (ms)'] >= xa) & (df_plot['Zeit (ms)'] <= xb)
        _df_int = df_plot[_mask]
        if len(_df_int) > 0:
            _zeichne_integral_flaeche(
                fig, _df_int['Zeit (ms)'].values, _df_int[active_sensor].values,
                yaxis=active_yaxis,
            )


# ---------------------------------------------------------------------------
# EXPORT: DIAGRAMM ALS PNG
# ---------------------------------------------------------------------------

def build_chart_png(
    df,
    sensor_namen: list[str],
    active_sensor: str,
    xa, xb, ya, yb, show_v_avg,
    t_vmax_start, y_vmax_start, t_vmax_ende, y_vmax_ende, has_vmax,
    t_vmin_start, y_vmin_start, t_vmin_ende, y_vmin_ende, has_vmin,
    t_amax_falling, y_amax_falling, has_amax_falling,
    t_amax_rising,  y_amax_rising,  has_amax_rising,
    show_rect_fit=False, show_rect_fit_top=True, rect_fit=None,
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

    _aktiv_einheit = kanal_einheit_map.get(active_sensor, 'µm')
    v_einheit, a_einheit, v_faktor, a_faktor = _ableit_info(_aktiv_einheit, zeit_einheit)

    _prim_einheit = next(iter(dict.fromkeys(kanal_einheit_map.get(n, 'µm') for n in sensor_namen)), 'µm')
    _prim_namen   = [n for n in sensor_namen if kanal_einheit_map.get(n, 'µm') == _prim_einheit] or sensor_namen
    y_max   = float(df[_prim_namen].max().max())
    y_min   = float(df[_prim_namen].min().min())
    y_range = [y_min, y_max + (y_max - y_min) * Y_PUFFER]

    velocity = acceleration = None
    if len(df) > 1:
        arr  = df[active_sensor].values
        dt_s = (df['Zeit (ms)'].iloc[1] - df['Zeit (ms)'].iloc[0]) / hz_faktor
        if show_velocity:
            roh = reader.berechne_sg_ableitung(arr, dt_s, window_length, 1)
            velocity = roh * v_faktor if roh is not None else None
        if show_acceleration:
            roh = reader.berechne_sg_ableitung(arr, dt_s, window_length_accel, 2)
            acceleration = roh * a_faktor if roh is not None else None

    _alle = alle_sensor_namen if alle_sensor_namen is not None else sensor_namen
    _kanal_farbe_map = {name: KANAL_FARBEN[_alle.index(name) if name in _alle else 0]
                        for name in sensor_namen}
    _kanal_bereiche: dict[str, tuple[float, float]] = {
        n: (float(df[n].min()), float(df[n].max())) for n in sensor_namen if n in df.columns
    }
    kanal_zu_yaxis, layout_yachsen, v_yaxis, a_yaxis, x_domain_end = _yachsen_layout(
        sensor_namen, kanal_einheit_map, y_range,
        show_velocity, velocity is not None,
        show_acceleration, acceleration is not None,
        v_einheit=v_einheit, a_einheit=a_einheit,
        kanal_farbe_map=_kanal_farbe_map,
        kanal_bereiche=_kanal_bereiche,
        kanal_ch_num=kanal_ch_num,
    )
    active_yaxis = kanal_zu_yaxis.get(active_sensor, 'y')

    export_fig = go.Figure()
    t_min = float(df['Zeit (ms)'].min())
    t_max = float(df['Zeit (ms)'].max())

    _baue_traces(
        export_fig, df, sensor_namen,
        kanal_zu_yaxis, active_sensor, active_yaxis,
        _alle,
        xa, xb, ya, yb, t_min, t_max,
        show_v_avg, rect_fit, show_rect_fit, show_rect_fit_top,
        has_vmax, t_vmax_start, y_vmax_start, t_vmax_ende, y_vmax_ende,
        has_vmin, t_vmin_start, y_vmin_start, t_vmin_ende, y_vmin_ende,
        has_amax_falling, t_amax_falling, y_amax_falling,
        has_amax_rising, t_amax_rising, y_amax_rising,
        sop_linien or [],
        show_velocity, velocity, v_yaxis,
        show_acceleration, acceleration, a_yaxis,
        show_integral,
    )

    export_fig.update_layout(
        xaxis_title=f"Zeit ({zeit_einheit})",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
        xaxis=dict(autorange=True, rangemode='nonnegative', domain=[0, x_domain_end],
                   showgrid=True, gridcolor='rgba(180,180,180,0.4)', gridwidth=1, nticks=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        **layout_yachsen,
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
