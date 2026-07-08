"""
Datei-Merger – kombiniert Kanäle aus mehreren Dateien GLEICHEN Dateityps
zu einer gemeinsamen Ausgabedatei.

Unterstützte Eingabeformate: CSV plain, Hubmessung TXT, Oszilloskop CSV, CSX.
CSX-Dateien werden als Oszilloskop-CSV geparst; Kanalnamen und Einheiten
werden aus dem eingebetteten JSON-Block (###CSX_SETTINGS_BEGIN###) übernommen.

Diese Seite ist Teil der Streamlit-Multipage-Navigation (Ordner `pages/`)
und läuft im selben Prozess wie `app.py`, ohne dass `app.py` selbst
verändert werden musste.

Workflow:
1. Dateityp wählen (muss für alle 4 Slots identisch sein).
2. In bis zu 4 Spalten je eine Datei hochladen.
3. Pro Datei werden die gefundenen Kanäle, die Sample-Anzahl und die
   Zeitbasis (Δt / Samplerate) angezeigt.
4. Gewünschte Kanäle ankreuzen (insgesamt max. 4, wie in der Hauptanwendung).
5. Kombinierte CSV (plain) herunterladen und in der Hauptanwendung
   als Dateityp "CSV plain" hochladen (Kopfzeilen überspringen = 0).
"""
from __future__ import annotations

import sys
import pathlib

import json

import numpy as np
import pandas as pd
import streamlit as st

# Lokales reader.py bevorzugen (wie in app.py)
_APP_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

import reader

MAX_KANAELE = 4   # entspricht N_KANÄLE in app.py

# Muss mit EINHEIT_OPTIONEN in app.py synchron bleiben
EINHEIT_OPTIONEN = ['µm', 'mm', 'm', 'V', 'mV', 'A', 'mA', 'N', 'kN', 'bar', 'Pa', '°C', '%']


def _fmt_de(value: float, nachkomma: int = 0) -> str:
    """Formatiert eine Zahl im deutschen Format (Punkt=Tausender, Komma=Dezimal)."""
    s = f"{value:,.{nachkomma}f}"
    # Python liefert z.B. '392,156.9' (en-US) -> nach de-DE tauschen
    s = s.replace(",", "").replace(".", ",").replace("", ".")
    return s


@st.cache_data(show_spinner=False)
def _lade_datei_gecacht(
    file_bytes: bytes,
    file_type: str,
    skip_rows: int,
) -> tuple[pd.DataFrame, float, tuple[str, ...], tuple[str, ...], tuple[str | None, ...]]:
    """Liest Datei ein und cached das Ergebnis; kein Re-Parse bei jedem Rerun.

    Gibt (raw_df, hz_faktor, messspalten, anzeige_namen, quell_einheiten) zurück:
    - messspalten:     Spaltennamen im raw_df (K1..Kn) – immer gültige df-Keys
    - anzeige_namen:   lesbare Namen aus dem Datei-Header (für UI und Export-Default)
    - quell_einheiten: Einheiten aus der Quelldatei (str) oder None falls unbekannt
    """
    if file_type == "CSX":
        csv_bytes, csx_settings = reader.split_csx(file_bytes)
        n_kanaele = reader.detect_kanal_count(csv_bytes, "Oszilloskop CSV", 0)
        kanal_namen_tmp = tuple(f"K{i}" for i in range(1, n_kanaele + 1))
        raw_df, hz_faktor = reader.load_raw(csv_bytes, "Oszilloskop CSV", 0, 0, kanal_namen_tmp)
        messspalten = tuple(c for c in raw_df.columns if c != "Zeit (s)")
        n = len(messspalten)
        if csx_settings:
            anzeige_namen = tuple(
                csx_settings.get(f'ch{i}_name', f'K{i}') for i in range(1, n + 1)
            )
            quell_einheiten = tuple(
                csx_settings.get(f'ch{i}_einheit') for i in range(1, n + 1)
            )
        else:
            anzeige_namen    = messspalten
            quell_einheiten  = tuple(None for _ in messspalten)
        return raw_df, hz_faktor, messspalten, anzeige_namen, quell_einheiten

    n_kanaele = reader.detect_kanal_count(file_bytes, file_type, skip_rows)
    kanal_namen_tmp = tuple(f"K{i}" for i in range(1, n_kanaele + 1))
    # reader.load_raw statt manuellem if/elif/else-Dispatch
    raw_df, hz_faktor = reader.load_raw(file_bytes, file_type, skip_rows, 0, kanal_namen_tmp)
    # Tatsächliche Messspalten aus dem geladenen DataFrame ableiten –
    # read_hubmessung_txt kürzt kanal_namen intern, wenn die Datei weniger
    # Spalten hat als detect_kanal_count zurückgegeben hat.
    messspalten = tuple(c for c in raw_df.columns if c != "Zeit (s)")
    # Lesbare Anzeigenamen: für CSV plain Header-Zeile nach skip_rows lesen
    if file_type == "CSV plain":
        peeked = reader.peek_csv_plain_kanalnames(file_bytes, len(messspalten), skip_rows)
        anzeige_namen = tuple(peeked) if peeked else messspalten
    else:
        anzeige_namen = messspalten
    quell_einheiten: tuple[str | None, ...] = tuple(None for _ in messspalten)
    return raw_df, hz_faktor, messspalten, anzeige_namen, quell_einheiten


st.set_page_config(layout="wide", page_title="Datei-Merger / Converter", page_icon="🔗")

st.title("🔗 Datei-Merger / Converter")
st.caption(
    "Kombiniert Kanäle aus mehreren Dateien **gleichen Dateityps** zu einer gemeinsamen Datei "
    "und konvertiert zwischen den Formaten **CSV plain**, **TXT (Hubmessung)** und **Oszilloskop CSV** – "
    "z. B. 4 Einzelmessungen mit je 1 Kanal zu einer 4-Kanal-Datei oder Hubmessung → Oszilloskop CSV."
)

file_type = st.radio(
    "Dateityp (für alle Dateien identisch)",
    ["CSV plain", "Hubmessung", "Oszilloskop CSV", "CSX"],
    horizontal=True,
    key="merger_file_type",
    help="Alle hochgeladenen Dateien müssen vom selben Dateityp sein.",
)
_file_ext = {"Hubmessung": ["txt"], "CSX": ["csx"]}.get(file_type, ["csv"])

st.divider()

# ---------------------------------------------------------------------------
# Schritt 1: Bis zu 4 Dateien hochladen und Kanäle anzeigen/auswählen
# ---------------------------------------------------------------------------
slot_data: list[dict] = []
cols = st.columns(4)

for slot_idx, col in enumerate(cols, start=1):
    with col:
        st.subheader(f"Datei {slot_idx}")
        info: dict = {"selected": []}

        skip_rows = 0
        if file_type == "CSV plain":
            skip_rows = st.number_input(
                "Kopfzeilen überspringen", min_value=0, step=1, value=0,
                key=f"merger_skip_{slot_idx}",
                help="Nur nötig, falls die Quelldatei Metadaten-Zeilen vor den Messwerten hat.",
            )

        f = st.file_uploader(
            f"Datei {slot_idx} wählen", type=_file_ext,
            key=f"merger_file_{slot_idx}",
            label_visibility="collapsed",
        )
        info["file"] = f

        if f is not None:
            try:
                file_bytes = f.getvalue()
                raw_df, hz_faktor, messspalten, anzeige_namen, quell_einheiten = _lade_datei_gecacht(
                    file_bytes, file_type, int(skip_rows)
                )

                if "Zeit (s)" in raw_df.columns:
                    zeit = raw_df["Zeit (s)"].values
                    n_samples = len(zeit)
                    if n_samples > 1:
                        dt = float(zeit[1] - zeit[0])   # Sekunden
                        sr_hz = 1.0 / dt if dt != 0.0 else float("nan")
                    else:
                        dt, sr_hz = float("nan"), float("nan")
                else:
                    n_samples = len(raw_df)
                    dt, sr_hz = None, None   # CSV plain: keine eigene Zeitspalte in der Datei

                info.update(
                    df=raw_df,
                    namen=messspalten,             # Df-Spaltennamen (K1..Kn) – für df[]-Zugriff
                    anzeige_namen=anzeige_namen,   # Lesbare Namen aus Datei-Header
                    quell_einheiten=quell_einheiten,
                    hz_faktor=hz_faktor,
                    n_samples=n_samples,
                    dt=dt,
                    sr_hz=sr_hz,
                )

                st.success(f"{len(messspalten)} Kanal/Kanäle erkannt")
                st.caption(f"📊 {_fmt_de(n_samples)} Samples")
                if dt is not None and not np.isnan(dt):
                    st.caption(f"⏱️ Δt = {_fmt_de(dt * 1e3, 4)} ms  (≈ {_fmt_de(sr_hz, 1)} Hz)")
                else:
                    st.caption("⏱️ Zeitbasis: synthetisch (globale Samplerate in Hauptanwendung)")

                st.markdown("**Kanäle auswählen:**")
                for ki, kname in enumerate(messspalten):
                    anzeige = anzeige_namen[ki] if ki < len(anzeige_namen) else kname
                    label   = anzeige.strip() or kname
                    eu      = quell_einheiten[ki] if ki < len(quell_einheiten) else None
                    if eu:
                        label = f"{label} [{eu}]"
                    checked = st.checkbox(label, key=f"merger_ch_{slot_idx}_{ki}")
                    if checked:
                        info["selected"].append(ki)

            except Exception as exc:
                info["error"] = str(exc)
                st.error(f"Fehler beim Einlesen: {exc}")

        slot_data.append(info)

st.divider()

# ---------------------------------------------------------------------------
# Schritt 2: Ausgewählte Kanäle zusammenführen
# ---------------------------------------------------------------------------
selected: list[dict] = []
for slot_idx, info in enumerate(slot_data, start=1):
    df = info.get("df")
    if df is None:
        continue
    messspalten      = info["namen"]
    anzeige_namen    = info.get("anzeige_namen", messspalten)
    quell_einheiten  = info.get("quell_einheiten", ())
    for ki in info.get("selected", []):
        kname   = messspalten[ki]   # immer gültiger df-Key
        anzeige = anzeige_namen[ki] if ki < len(anzeige_namen) else kname
        einheit = quell_einheiten[ki] if ki < len(quell_einheiten) else None
        selected.append({
            "label":     anzeige.strip() or f"Datei{slot_idx}_K{ki + 1}",
            "slot_idx":  slot_idx,
            "kname":     kname,          # z.B. "K2" – Slot-Kanal-Bezeichner
            "array":     df[kname].values,
            "n":         len(df),
            "dt":        info.get("dt"),
            "hz_faktor": info.get("hz_faktor", 1000.0),
            "quelle":    info["file"].name,
            "einheit":   einheit,
        })

n_selected = len(selected)

if n_selected == 0:
    st.info("Bitte in mindestens einer Datei einen Kanal ankreuzen.")
elif n_selected > MAX_KANAELE:
    st.error(
        f"Es sind maximal {MAX_KANAELE} Kanäle erlaubt – aktuell sind {n_selected} "
        f"ausgewählt. Bitte {n_selected - MAX_KANAELE} Kanal/Kanäle abwählen."
    )
else:
    # Ausgabe-Kanalnamen zuerst festlegen (vereinfacht die Zuordnung) – alle
    # nachfolgenden Anzeigen (Übersichtstabelle, Füllwerte, Vorschau) verwenden
    # ab hier den neu vergebenen Namen; ohne Eingabe wird K1..K4 verwendet.
    st.markdown("**Ausgabe-Kanalnamen (optional anpassen):**")
    out_names: list[str] = []
    name_cols = st.columns(n_selected)
    for i, (c, ncol) in enumerate(zip(selected, name_cols)):
        with ncol:
            _field_label = f"Datei {c['slot_idx']} {c['kname']}"
            name = st.text_input(
                _field_label, value=c["label"], key=f"merger_outname_{i}",
            )
            out_names.append(name.strip() or f"K{i + 1}")

    # ------------------------------------------------------------------
    # Sampleraten angleichen (lineare Interpolation via numpy.interp)
    # Nur sinnvoll/möglich, wenn für ALLE gewählten Kanäle eine Zeitbasis
    # bekannt ist (Hubmessung / Oszilloskop CSV) UND sich die Δt-Werte
    # tatsächlich unterscheiden. Die Original-Arrays bleiben unangetastet;
    # das Resample-Ergebnis wird separat vorgehalten, damit die Übersichts-
    # tabelle Original- und resamplete Sample-Anzahl nebeneinander zeigen kann.
    # ------------------------------------------------------------------
    st.session_state.setdefault("merger_resample_active", False)

    _dts_bekannt = [c["dt"] for c in selected if c["dt"] is not None and not np.isnan(c["dt"])]
    _distinct_dts = {round(d, 6) for d in _dts_bekannt}
    resample_moeglich = len(_dts_bekannt) == n_selected and len(_distinct_dts) > 1
    target_dt = min(_dts_bekannt) if _dts_bekannt else None
    # hz_faktor des Kanals mit kleinstem Δt – für korrekte Hz-Anzeige im Resample-Ergebnis
    hz_faktor_target = next(
        (c["hz_faktor"] for c in selected
         if c["dt"] is not None and target_dt is not None and abs(c["dt"] - target_dt) < 1e-9),
        1000.0,
    )

    resample_arrays: list[np.ndarray | None] = [None] * n_selected
    resample_n: list[int | None] = [None] * n_selected

    if resample_moeglich:
        _aktiv = st.session_state["merger_resample_active"]
        _btn_label = (
            "↩️ Sampleraten-Angleichung zurücksetzen"
            if _aktiv else
            f"🔁 Sampleraten angleichen (lineare Interpolation auf Δt = {_fmt_de(target_dt * 1e3, 4)} ms)"
        )
        if st.button(_btn_label, key="merger_resample_btn", width="stretch",
                     help=(
                         "Interpoliert alle Kanäle linear (numpy.interp) auf die feinste "
                         "vorhandene Zeitbasis (kleinstes Δt) und ersetzt so die fehlenden "
                         "Samples der gröber abgetasteten Kanäle."
                     )):
            st.session_state["merger_resample_active"] = not _aktiv
            st.rerun()

        if st.session_state["merger_resample_active"]:
            try:
                for i, c in enumerate(selected):
                    dt_i, n_i = c["dt"], c["n"]
                    if abs(dt_i - target_dt) < 1e-12:
                        resample_arrays[i] = c["array"]
                        resample_n[i] = n_i
                        continue
                    t_orig = np.arange(n_i, dtype=np.float64) * dt_i
                    duration = t_orig[-1] if n_i > 1 else 0.0
                    n_new = int(round(duration / target_dt)) + 1
                    t_new = np.arange(n_new, dtype=np.float64) * target_dt
                    # np.interp: lineare Interpolation, füllt die fehlenden Zwischenwerte
                    resample_arrays[i] = np.interp(t_new, t_orig, c["array"])
                    resample_n[i] = n_new
                st.success(
                    f"Sampleraten linear interpoliert (numpy.interp) auf Δt = "
                    f"{_fmt_de(target_dt * 1e3, 4)} ms "
                    f"(≈ {_fmt_de(1.0 / target_dt, 1)} Hz)."
                )
            except Exception as exc:
                st.error(f"Resampling fehlgeschlagen: {exc}")
                st.session_state["merger_resample_active"] = False
    elif n_selected > 1 and not _dts_bekannt:
        st.caption(
            "ℹ️ Sampleraten-Angleichung nicht verfügbar: Dateityp 'CSV plain' hat keine "
            "eigene Zeitbasis je Datei (nur Hubmessung / Oszilloskop CSV unterstützt)."
        )

    resample_active = st.session_state["merger_resample_active"] and resample_moeglich

    # Effektive Arrays/Längen/Δt für alle nachfolgenden Schritte (Tabelle,
    # Längenanpassung, Ausgabe) – nutzt resamplete Werte falls aktiv.
    eff_arrays = [resample_arrays[i] if resample_active and resample_arrays[i] is not None
                  else c["array"] for i, c in enumerate(selected)]
    eff_n = [resample_n[i] if resample_active and resample_n[i] is not None
             else c["n"] for i, c in enumerate(selected)]
    eff_dt = [target_dt if resample_active else c["dt"] for c in selected]

    st.subheader(f"Ausgewählte Kanäle: {n_selected} / {MAX_KANAELE}")

    _uebersicht = pd.DataFrame([
        {
            "Kanalname": out_names[i],
            "Quelldatei": c["quelle"],
            "Kanal": c["label"],
            "Samples": c["n"],
            "dT (ms)": f"{c['dt'] * 1e3:.4g}" if c["dt"] is not None and not np.isnan(c["dt"]) else "—",
            "Resampled (n)": resample_n[i] if resample_n[i] is not None else "—",
            "new dt (ms)": _fmt_de(target_dt * 1e3, 4) if resample_n[i] is not None else "—",
        }
        for i, c in enumerate(selected)
    ])
    st.table(_uebersicht)

    lengths = set(eff_n)
    min_len = min(lengths)
    max_len = max(lengths)

    fill_mode = False
    fill_values: list[float] = [0.0] * n_selected
    if len(lengths) > 1:
        st.warning(
            f"Die gewählten Kanäle haben unterschiedliche Sample-Anzahlen ({sorted(lengths)})."
        )
        laenge_modus = st.radio(
            "Längenanpassung",
            ["Auf kürzeste Länge kürzen", "Mit Füllwert auf längste Länge auffüllen"],
            key="merger_laenge_modus",
            help=(
                f"Kürzen: alle Kanäle werden auf {min_len} Samples gekürzt. "
                f"Auffüllen: kürzere Kanäle werden mit einem festen Wert auf {max_len} Samples "
                "verlängert (Werte werden am Ende angehängt)."
            ),
        )
        fill_mode = laenge_modus.startswith("Mit Füllwert")

        if fill_mode:
            st.markdown("**Füllwert je Kanal (nur für Kanäle kürzer als die längste Länge):**")
            fill_cols = st.columns(n_selected)
            for i, (n_i, fcol) in enumerate(zip(eff_n, fill_cols)):
                with fcol:
                    if n_i < max_len:
                        fill_values[i] = st.number_input(
                            f"Füllwert {out_names[i]}",
                            value=0.0,
                            key=f"merger_fill_{i}",
                            help=(
                                f"{out_names[i]}: {n_i} von {max_len} Samples - "
                                f"fehlende {max_len - n_i} Samples werden mit diesem Wert aufgefüllt."
                            ),
                        )
                    else:
                        st.caption(f"{out_names[i]}: vollständig ({n_i} Samples)")

    dts = {round(d, 6) for d in eff_dt if d is not None and not np.isnan(d)}
    if len(dts) > 1:
        st.warning(
            f"⚠️ Die gewählten Kanäle haben unterschiedliche Zeitbasen (Δt: {sorted(dts)} ms). "
            "Es findet KEIN Resampling statt – die Kanäle werden unverändert nebeneinandergelegt. "
            "Das Ergebnis kann zeitlich verschoben sein, wenn die Quell-Samplerate abweicht. "
            "Nutze bei Bedarf den Button 'Sampleraten angleichen' oben."
        )

    out_len = max_len if fill_mode else min_len

    def _build_column(arr: np.ndarray, fill_value: float) -> np.ndarray:
        if len(arr) >= out_len:
            return arr[:out_len]
        # Kürzer als out_len -> mit Füllwert auf out_len verlängern
        padding = np.full(out_len - len(arr), fill_value, dtype=np.float64)
        return np.concatenate([arr, padding])

    out_df = pd.DataFrame({
        name: _build_column(arr, fv)
        for name, arr, fv in zip(out_names, eff_arrays, fill_values)
    })

    st.markdown("**Vorschau (erste 20 Zeilen):**")
    st.dataframe(out_df.head(20), width="stretch")

    st.divider()
    st.markdown("**Export**")

    # --- Gemeinsames Δt für alle zeitachsenbasierten Formate ---
    _dts_eff = [d for d in eff_dt if d is not None and not np.isnan(d)]
    if _dts_eff:
        _dt_export       = float(min(_dts_eff))   # Sekunden
        _hz_faktor_export = hz_faktor_target
        _zeit_einheit_disp = {1000.0: 'ms', 1_000_000.0: 'µs', 1_000_000_000.0: 'ns', 1.0: 's'}.get(
            float(_hz_faktor_export), 'ms'
        )
        _dt_s_export = _dt_export   # bereits Sekunden
        st.caption(
            f"⏱️ Δt = {_fmt_de(_dt_export * _hz_faktor_export, 6)} {_zeit_einheit_disp} "
            f"(≈ {_fmt_de(1.0 / _dt_export, 1)} Hz, aus Quelldateien)"
        )
    else:
        _dt_export = float(st.number_input(
            "Δt (ms) für Zeitachse",
            min_value=1e-9, value=0.005, format="%.6f",
            key="merger_export_dt",
            help="CSV plain hat keine eigene Zeitbasis – Zeitschritt hier manuell eingeben.",
        ))
        _hz_faktor_export = 1000.0   # manuelle Eingabe immer in ms
        _dt_s_export = _dt_export / 1000.0   # ms → Sekunden

    # --- Einheiten pro Kanal (Oszilloskop CSV-Header) ---
    st.markdown("**Einheiten** (für Oszilloskop- und CSX-Export):")
    _einheit_spalten = st.columns(n_selected)
    _einheiten_export: list[str] = []
    for _i, _ecol in enumerate(_einheit_spalten):
        with _ecol:
            _quell_eu = selected[_i].get("einheit")
            _eu_key   = f"merger_osc_einheit_{_i}"
            _eu_idx   = (
                EINHEIT_OPTIONEN.index(_quell_eu)
                if _quell_eu in EINHEIT_OPTIONEN and _eu_key not in st.session_state
                else 0
            )
            _einheiten_export.append(
                st.selectbox(
                    out_names[_i],
                    options=EINHEIT_OPTIONEN,
                    index=_eu_idx,
                    key=_eu_key,
                )
            )

    # --- Oszilloskop-CSV-Bytes vorab bauen (wird auch vom CSX-Export benötigt) ---
    _osc_header = "x-axis," + ",".join(str(_i + 1) for _i in range(n_selected))
    _osc_units  = "second," + ",".join(_einheiten_export)
    _zeit_s     = np.arange(out_len, dtype=np.float64) * _dt_s_export
    _val_lines  = out_df.to_csv(index=False, header=False, float_format="%.6f").splitlines()
    _osc_daten  = [f"{_t:.9e},{_v}" for _t, _v in zip(_zeit_s, _val_lines)]
    _osc_bytes  = "\n".join([_osc_header, _osc_units] + _osc_daten).encode("utf-8")

    dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)

    # --- CSV plain ---
    with dl_col1:
        _csv_bytes = out_df.to_csv(index=False, float_format="%.6f").encode("utf-8")
        st.download_button(
            "💾 CSV (plain)",
            data=_csv_bytes,
            file_name="kombiniert_plain.csv",
            mime="text/csv",
            width="stretch",
        )
        st.caption("Dateityp 'CSV plain', Kopfzeilen = 0.")

    # --- TXT Hubmessung ---
    with dl_col2:
        _dt_ms_export = _dt_s_export * 1000.0   # Sekunden → ms für TXT-Zeitachse
        _zeit_ms = np.arange(out_len, dtype=np.float64) * _dt_ms_export
        _txt_df  = pd.DataFrame({"Time (ms)": _zeit_ms})
        for _col in out_df.columns:
            _txt_df[_col] = out_df[_col].values
        # Raw Data: Marker + TAB-getrennt + Komma-Dezimal
        _txt_bytes = (
            "Raw Data:\n"
            + _txt_df.to_csv(index=False, sep="\t", decimal=",", float_format="%.6f")
        ).encode("utf-8")
        st.download_button(
            "💾 TXT (Hubmessung)",
            data=_txt_bytes,
            file_name="kombiniert_hubmessung.txt",
            mime="text/plain",
            width="stretch",
        )
        st.caption("Dateityp 'Hubmessung'.")

    # --- Oszilloskop CSV ---
    with dl_col3:
        st.download_button(
            "💾 Oszilloskop CSV",
            data=_osc_bytes,
            file_name="kombiniert_oszilloskop.csv",
            mime="text/csv",
            width="stretch",
        )
        st.caption("Dateityp 'Oszilloskop CSV'.")

    # --- CSX (Oszilloskop CSV + eingebettete Kanal-Einstellungen) ---
    with dl_col4:
        # Dropdown garantiert bereits gültige Einheiten; auf MAX_KANAELE auffüllen
        _einheiten_csx = list(_einheiten_export)
        while len(_einheiten_csx) < MAX_KANAELE:
            _einheiten_csx.append('µm')

        _sr_hz     = (1.0 / _dt_s_export) if _dt_s_export > 0 else 1000.0
        _total_s   = (out_len - 1) * _dt_s_export
        # Zeiteinheit bestimmen wie sie die Haupt-App beim Import wählen würde
        _hz_f_disp, _ = reader.wahl_zeiteinheit(_total_s) if _total_s > 0 else (1000.0, 'ms')
        _xb_csx    = round(_total_s * _hz_f_disp, 3)
        _csx_settings: dict = {
            'file_type_radio':      'Oszilloskop CSV',
            'max_samples':          0,
            'skip_rows':            0,
            'active_sensor_name':   out_names[0] if out_names else '',
            'sample_rate':          _sr_hz,
            'sample_rate_unit':     'Hz',
            'sample_rate_unit_toggle': False,
            # Cursor und Crop auf Gesamtbereich zurücksetzen
            'xa':          0.0,
            'xb':          _xb_csx,
            'crop_start':  None,
            'crop_end':    None,
        }
        for _ci in range(1, MAX_KANAELE + 1):
            _in_export = _ci <= n_selected
            _csx_settings[f'ch{_ci}_name']    = out_names[_ci - 1] if _in_export else ''
            _csx_settings[f'ch{_ci}_einheit'] = _einheiten_csx[_ci - 1]
            _csx_settings[f'osc_skale_{_ci}'] = 1.0
            _csx_settings[f'off{_ci}']        = 0.0
            _csx_settings[f'off{_ci}_slider'] = 0.0
            _csx_settings[f'x_off{_ci}']     = 0.0
            _csx_settings[f'show_ch{_ci}']   = _in_export
            _csx_settings[f'ch{_ci}_sg_en']  = False
            _csx_settings[f'ch{_ci}_sg_win'] = 5
            _csx_settings[f'ch{_ci}_ymin']   = 0.0
            _csx_settings[f'ch{_ci}_ymax']   = 0.0

        _csx_bytes = (
            _osc_bytes
            + b"\n###CSX_SETTINGS_BEGIN###\n"
            + json.dumps(_csx_settings, ensure_ascii=False).encode('utf-8')
            + b"\n###CSX_SETTINGS_END###\n"
        )
        st.download_button(
            "💾 CSX",
            data=_csx_bytes,
            file_name="kombiniert.csx",
            mime="text/plain",
            width="stretch",
        )
        st.caption("Oszi CSV + Kanaleinstellungen, direkt in Haupt-App laden.")
