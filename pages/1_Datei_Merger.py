"""
Datei-Merger – kombiniert Kanäle aus mehreren Dateien GLEICHEN Dateityps
zu einer gemeinsamen CSV (plain) Datei.

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

import numpy as np
import pandas as pd
import streamlit as st

# Lokales reader.py bevorzugen (wie in app.py)
_APP_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

import reader

MAX_KANAELE = 4   # entspricht N_KANÄLE in app.py


def _fmt_de(value: float, nachkomma: int = 0) -> str:
    """Formatiert eine Zahl im deutschen Format (Punkt=Tausender, Komma=Dezimal)."""
    s = f"{value:,.{nachkomma}f}"
    # Python liefert z.B. '392,156.9' (en-US) -> nach de-DE tauschen
    s = s.replace(",", "\u0001").replace(".", ",").replace("\u0001", ".")
    return s

st.set_page_config(layout="wide", page_title="Datei-Merger", page_icon="🔗")

st.title("🔗 Datei-Merger")
st.caption(
    "Kombiniert Kanäle aus mehreren Dateien **gleichen Dateityps** zu einer "
    "gemeinsamen CSV (plain)-Datei – z. B. 4 Einzelmessungen mit je 1 Kanal "
    "zu einer 4-Kanal-Datei. Die erzeugte Datei kann anschließend ganz normal "
    "in der Hauptanwendung hochgeladen werden."
)

file_type = st.radio(
    "Dateityp (für alle Dateien identisch)",
    ["CSV plain", "Hubmessung", "Oszilloskop CSV"],
    horizontal=True,
    key="merger_file_type",
    help="Alle hochgeladenen Dateien müssen vom selben Dateityp sein.",
)
_file_ext = ["txt"] if file_type == "Hubmessung" else ["csv"]

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
                n_kanaele_datei = reader.detect_kanal_count(file_bytes, file_type, skip_rows)

                if file_type == "CSV plain":
                    peeked = reader.peek_csv_plain_kanalnames(file_bytes, n_kanaele_datei)
                    kanal_namen = tuple(peeked) if peeked else tuple(
                        f"K{i}" for i in range(1, n_kanaele_datei + 1)
                    )
                else:
                    kanal_namen = tuple(f"K{i}" for i in range(1, n_kanaele_datei + 1))

                if file_type == "Hubmessung":
                    raw_df = reader.read_hubmessung_txt(file_bytes, 0, kanal_namen)
                    hz_faktor = 1000.0
                elif file_type == "Oszilloskop CSV":
                    raw_df, hz_faktor = reader.read_oszilloskop_csv(
                        file_bytes, 0, kanal_namen, tuple(1.0 for _ in kanal_namen)
                    )
                else:
                    raw_df = reader.read_csv_plain(file_bytes, skip_rows, 0, kanal_namen)
                    hz_faktor = 1000.0

                if "Zeit (ms)" in raw_df.columns:
                    zeit = raw_df["Zeit (ms)"].values
                    n_samples = len(zeit)
                    if n_samples > 1:
                        dt = float(zeit[1] - zeit[0])
                        sr_hz = hz_faktor / dt if dt not in (0, None) else float("nan")
                    else:
                        dt, sr_hz = float("nan"), float("nan")
                else:
                    n_samples = len(raw_df)
                    dt, sr_hz = None, None   # CSV plain: keine eigene Zeitspalte in der Datei

                info.update(df=raw_df, namen=kanal_namen, hz_faktor=hz_faktor,
                            n_samples=n_samples, dt=dt, sr_hz=sr_hz)

                st.success(f"{len(kanal_namen)} Kanal/Kanäle erkannt")
                st.caption(f"📊 {_fmt_de(n_samples)} Samples")
                if dt is not None and not np.isnan(dt):
                    st.caption(f"⏱️ Δt = {_fmt_de(dt, 4)} ms  (≈ {_fmt_de(sr_hz, 1)} Hz)")
                else:
                    st.caption("⏱️ Zeitbasis: synthetisch (globale Samplerate in Hauptanwendung)")

                st.markdown("**Kanäle auswählen:**")
                for ki, kname in enumerate(kanal_namen):
                    label = kname.strip() if kname and kname.strip() else f"Kanal {ki + 1}"
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
    for ki in info.get("selected", []):
        kname = info["namen"][ki]
        selected.append({
            "label": kname.strip() if kname and kname.strip() else f"Datei{slot_idx}_K{ki + 1}",
            "array": df[kname].values,
            "n": len(df[kname].values),
            "dt": info.get("dt"),
            "quelle": info["file"].name,
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
            name = st.text_input(
                f"Name Kanal {i + 1}", value=c["label"], key=f"merger_outname_{i}",
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
    if "merger_resample_active" not in st.session_state:
        st.session_state["merger_resample_active"] = False

    _dts_bekannt = [c["dt"] for c in selected if c["dt"] is not None and not np.isnan(c["dt"])]
    _distinct_dts = {round(d, 6) for d in _dts_bekannt}
    resample_moeglich = len(_dts_bekannt) == n_selected and len(_distinct_dts) > 1
    target_dt = min(_dts_bekannt) if _dts_bekannt else None

    resample_arrays: list[np.ndarray | None] = [None] * n_selected
    resample_n: list[int | None] = [None] * n_selected

    if resample_moeglich:
        _aktiv = st.session_state["merger_resample_active"]
        _btn_label = (
            "↩️ Sampleraten-Angleichung zurücksetzen"
            if _aktiv else
            f"🔁 Sampleraten angleichen (lineare Interpolation auf Δt = {_fmt_de(target_dt, 4)} ms)"
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
                f"{_fmt_de(target_dt, 4)} ms (≈ {_fmt_de(1000.0 / target_dt, 1)} Hz)."
            )
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
            "dT (ms)": f"{c['dt']:.4g}" if c["dt"] is not None and not np.isnan(c["dt"]) else "—",
            "Resampled (n)": resample_n[i] if resample_n[i] is not None else "—",
            "new dt (ms)": _fmt_de(target_dt, 4) if resample_n[i] is not None else "—",
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

    csv_bytes = out_df.to_csv(index=False, float_format="%.6f").encode("utf-8")
    st.download_button(
        "💾 Kombinierte CSV (plain) herunterladen",
        data=csv_bytes,
        file_name="kombiniert_plain.csv",
        mime="text/csv",
        width="stretch",
    )
    st.caption(
        "ℹ️ Beim Hochladen in der Hauptanwendung: Dateityp 'CSV plain', "
        "'Kopfzeilen überspringen' = 0 – die Kanalnamen werden automatisch aus der "
        "ersten Zeile übernommen. Die Abtastrate bitte passend zur ursprünglichen "
        "Zeitbasis der Quelldateien manuell einstellen (siehe Delta-t-Spalte oben)."
    )
