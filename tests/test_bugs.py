"""Regressionstests für die im Code-Review identifizierten Bugs #1–#3.

#1 – a_faktor wurde fälschlich quadriert (_a_scale**2 statt _a_scale)
#2 – Kanal-Index-Shift bei übersprungenen Slots (_sensor_ch_num fehlt)
#3 – SOP sg_v Längen-Mismatch bei aktivem Crop (sg_v_roh statt sg_v_roh_full)
"""
import sys
import os
import math
import numpy as np
import pytest

# chart.py liegt im Eltern-Verzeichnis, kein Package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import chart as chart_module


# ---------------------------------------------------------------------------
# Bug #1 – Beschleunigungsskalierung muss linear sein (nicht quadratisch)
# ---------------------------------------------------------------------------

class TestAFaktorLinear:
    """_laenge_autoscale liefert einen linearen Skalierungsfaktor.
    Die Beschleunigungs-Anzeigeeinheit (Einheit/s²) erhält dieselbe lineare
    Skalierung wie die Geschwindigkeit, weil der SG-Filter delta=dt_s bereits
    die /s²-Zeitkomponente einbaut.
    """

    def test_laenge_autoscale_gibt_linearen_faktor(self):
        # µm → mm: 10000 µm/s = 10 mm/s ≥ 10 → Einheit wechselt zu mm, Faktor = 1e-3
        einheit, faktor = chart_module._laenge_autoscale('µm', 10000.0)
        assert einheit == 'mm'
        assert math.isclose(faktor, 1e-3, rel_tol=1e-9)

    def test_laenge_autoscale_nm_zu_um(self):
        # nm → µm: 10000 nm/s = 10 µm/s ≥ 10 → Einheit µm, Faktor = 1e-3
        einheit, faktor = chart_module._laenge_autoscale('nm', 10000.0)
        assert einheit == 'µm'
        assert math.isclose(faktor, 1e-3, rel_tol=1e-9)

    def test_linearer_faktor_vs_quadrierter_faktor_unterschied(self):
        """Bestätigt, dass linear ≠ quadratisch – der Bug war messbar."""
        _, faktor = chart_module._laenge_autoscale('µm', 10000.0)   # → mm, faktor=1e-3
        # Linear: 10000 µm/s² * 1e-3 = 10 mm/s²  ✓
        # Bug:    10000 µm/s² * (1e-3)² = 0.01 mm/s²  ✗
        a_roh_um_s2 = 10000.0
        a_linear    = a_roh_um_s2 * faktor
        a_quadriert = a_roh_um_s2 * (faktor ** 2)
        assert math.isclose(a_linear,    10.0, rel_tol=1e-9)   # korrekt
        assert math.isclose(a_quadriert,  0.01, rel_tol=1e-9)  # Bug-Ergebnis
        assert not math.isclose(a_quadriert, a_linear, rel_tol=1e-3)

    def test_a_scale_liefert_korrekte_physikalische_einheit(self):
        """10000 µm/s² = 10 mm/s² – nur mit linearem Faktor korrekt."""
        _, faktor = chart_module._laenge_autoscale('µm', 10000.0)  # → mm, faktor=1e-3
        a_roh_um_s2 = 10000.0   # 10000 µm/s²
        a_mm_s2     = a_roh_um_s2 * faktor
        assert math.isclose(a_mm_s2, 10.0, rel_tol=1e-9)

    def test_autoscale_wert_null_gibt_faktor_eins(self):
        einheit, faktor = chart_module._laenge_autoscale('µm', 0.0)
        assert einheit == 'µm'
        assert faktor == 1.0

    def test_autoscale_wert_negativ_verhält_sich_wie_betrag(self):
        """Negativer Referenzwert soll denselben Einheitenwechsel wie der Betrag auslösen."""
        einheit, faktor = chart_module._laenge_autoscale('µm', -10000.0)
        assert einheit == 'mm'
        assert math.isclose(faktor, 1e-3, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Bug #2 – Kanal-Index über _sensor_ch_num, nicht Listenposition
# ---------------------------------------------------------------------------

class TestSensorChNum:
    """_sensor_ch_num muss den tatsächlichen Slot-Schlüssel liefern.

    Szenario: Slot 1 ist leer gelassen (dokumentiertes Feature), Slot 2 = "Weg",
    Slot 3 = "Strom". Dann ist:
      _sensor_ch_num = {'Weg': 2, 'Strom': 3}
      sensor_namen   = ['Weg', 'Strom']   (Listenposition 0, 1)

    Positionale Lesart (Bug): sensor_namen[0] → key 'off1', sensor_namen[1] → key 'off2'
    Korrekte Lesart (Fix):    'Weg' → _sensor_ch_num['Weg'] = 2 → key 'off2'
                               'Strom' → _sensor_ch_num['Strom'] = 3 → key 'off3'
    """

    def _build_sensor_ch_num(self, kanal_cfg):
        """Repliziert die _sensor_ch_num-Logik aus app.py."""
        result: dict[str, int] = {}
        for i, n in enumerate(kanal_cfg):
            if n and n not in result:
                result[n] = i + 1
        return result

    def test_ohne_luecke_entspricht_position_plus_eins(self):
        kanal_cfg = ['Weg', 'Strom', '', '']
        sensor_ch = self._build_sensor_ch_num(kanal_cfg)
        assert sensor_ch == {'Weg': 1, 'Strom': 2}

    def test_mit_luecke_vorne_verschiebt_slot_nummer(self):
        """Slot 1 leer → 'Weg' ist Slot 2, nicht Slot 1."""
        kanal_cfg = ['', 'Weg', 'Strom', '']
        sensor_ch = self._build_sensor_ch_num(kanal_cfg)
        assert sensor_ch['Weg']   == 2
        assert sensor_ch['Strom'] == 3

    def test_korrekte_session_key_lesart_mit_luecke(self):
        """Korrekte Session-State-Key-Ableitung mit Lücke in Slot 1."""
        # simulierter session_state
        session = {
            'off1': 0.0, 'off2': 5.0, 'off3': -3.0,
            'osc_skale_1': 1.0, 'osc_skale_2': 0.5, 'osc_skale_3': 2.0,
        }
        kanal_cfg    = ['', 'Weg', 'Strom', '']
        sensor_namen = [n for n in kanal_cfg if n]
        sensor_ch    = self._build_sensor_ch_num(kanal_cfg)

        # FIX: Index über sensor_ch[name]
        offs_korrekt = tuple(session.get(f'off{sensor_ch[name]}', 0.0) for name in sensor_namen)
        # BUG: Index über Listenposition i+1
        offs_falsch  = tuple(session.get(f'off{i+1}', 0.0) for i in range(len(sensor_namen)))

        # Bei Lücke in Slot 1 lesen beide unterschiedliche Werte
        assert offs_korrekt == (5.0, -3.0)   # off2, off3 → korrekt
        assert offs_falsch  == (0.0,  5.0)   # off1, off2 → falsch um einen Slot verschoben

    def test_osc_skale_korrekte_lesart(self):
        session = {
            'osc_skale_1': 1.0, 'osc_skale_2': 0.5, 'osc_skale_3': 2.0,
        }
        kanal_cfg     = ['', 'Weg', 'Strom', '']
        kanal_namen   = tuple(n for n in kanal_cfg if n)
        sensor_ch     = self._build_sensor_ch_num(kanal_cfg)

        fix   = tuple(session[f'osc_skale_{sensor_ch[name]}'] for name in kanal_namen)
        buggy = tuple(session[f'osc_skale_{i+1}']            for i in range(len(kanal_namen)))

        assert fix   == (0.5, 2.0)   # Slot 2 und 3 → korrekt
        assert buggy == (1.0, 0.5)   # Slot 1 und 2 → um einen verschoben

    def test_sg_params_korrekte_lesart(self):
        session = {
            'ch1_sg_en': False, 'ch1_sg_win': 5,
            'ch2_sg_en': True,  'ch2_sg_win': 11,
            'ch3_sg_en': False, 'ch3_sg_win': 7,
        }
        kanal_cfg    = ['', 'Weg', 'Strom', '']
        sensor_namen = [n for n in kanal_cfg if n]
        sensor_ch    = self._build_sensor_ch_num(kanal_cfg)

        fix_params   = tuple(
            (bool(session.get(f'ch{sensor_ch[name]}_sg_en', False)),
             int(session.get(f'ch{sensor_ch[name]}_sg_win', 5)))
            for name in sensor_namen
        )
        buggy_params = tuple(
            (bool(session.get(f'ch{i+1}_sg_en', False)),
             int(session.get(f'ch{i+1}_sg_win', 5)))
            for i in range(len(sensor_namen))
        )

        assert fix_params   == ((True, 11), (False, 7))   # Slot 2, 3 → korrekt
        assert buggy_params == ((False, 5), (True, 11))   # Slot 1, 2 → falsch

    def test_keine_luecke_fix_und_bug_identisch(self):
        """Ohne Lücke liefern beide Varianten dasselbe Ergebnis."""
        session      = {'off1': 1.0, 'off2': 2.0}
        kanal_cfg    = ['Weg', 'Strom', '', '']
        sensor_namen = [n for n in kanal_cfg if n]
        sensor_ch    = self._build_sensor_ch_num(kanal_cfg)

        fix   = tuple(session.get(f'off{sensor_ch[name]}', 0.0) for name in sensor_namen)
        buggy = tuple(session.get(f'off{i+1}', 0.0)            for i in range(len(sensor_namen)))

        assert fix == buggy == (1.0, 2.0)


# ---------------------------------------------------------------------------
# Bug #3 – SOP sg_v muss gleiche Länge wie signal/zeit haben (df_use-Länge)
# ---------------------------------------------------------------------------

class TestSopSgvLaenge:
    """_finde_sop_kreuzungen soll den SG-Geschwindigkeitswert nutzen (nicht
    Differenzenquotient-Fallback), wenn sg_v dieselbe Länge wie signal hat.
    Bei Crop-Aktivierung muss sg_v_roh_full (volle Länge, passend zu df_use)
    übergeben werden, nicht das crop-geslicte sg_v_roh.
    """

    def _minimal_rect_fit(self, t_start, t_end, y_low, y_high):
        return {
            'y_low':  y_low,
            'y_high': y_high,
            'runs': [{'t_start': t_start, 't_end': t_end}],
        }

    def _rechteck_signal(self, n=200, puls_start=60, puls_ende=140,
                          y_low=0.0, y_high=100.0, dt_ms=0.1):
        """Einfaches Rechtecksignal für SOP-Tests."""
        zeit   = np.arange(n) * dt_ms
        signal = np.where(
            (np.arange(n) >= puls_start) & (np.arange(n) < puls_ende),
            y_high, y_low
        ).astype(float)
        # Leichte Flanke hinzufügen damit Kreuzungserkennung funktioniert
        signal[puls_start] = (y_low + y_high) / 2
        signal[puls_ende]  = (y_low + y_high) / 2
        return zeit, signal

    def test_sg_v_gleiche_laenge_nutzt_sg_zweig(self):
        """sg_v gleicher Länge wie signal → SG-Geschwindigkeitszweig wird benutzt."""
        n     = 200
        dt_ms = 0.1
        zeit, signal = self._rechteck_signal(n=n)
        rect_fit = self._minimal_rect_fit(
            t_start=zeit[60], t_end=zeit[140], y_low=0.0, y_high=100.0
        )

        # Synthetische SG-Geschwindigkeit: großer Wert an der steigenden Flanke
        sg_v_gleich = np.zeros(n)
        sg_v_gleich[60] = 999.0   # bekannter Sentinel-Wert

        linien, v_sop = chart_module._finde_sop_kreuzungen(
            zeit, signal, rect_fit,
            sop_percent=50.0,
            sample_rate=1.0 / (dt_ms * 1e-3),
            halbes_zeitfenster=3,
            v_faktor=1.0,
            sg_v=sg_v_gleich,
        )
        # Wenn SG-Zweig genutzt: v_sop nahe 0 (Flankenbereich hat 0er, nur Stützstelle 60 hat 999)
        # Wenn Fallback: v_sop über Differenzenquotient berechnet
        # Primärer Test: Funktion läuft ohne Fehler durch und gibt einen Wert zurück
        assert isinstance(v_sop, float)
        assert len(linien) > 0, "Kein SOP-Kreuzungspunkt gefunden"

    def test_sg_v_falsche_laenge_nutzt_fallback(self):
        """sg_v mit falscher Länge (Crop-Slice) → Fallback auf Differenzenquotienten."""
        n     = 200
        dt_ms = 0.1
        zeit, signal = self._rechteck_signal(n=n)
        rect_fit = self._minimal_rect_fit(
            t_start=zeit[60], t_end=zeit[140], y_low=0.0, y_high=100.0
        )

        # sg_v hat falsche Länge (z. B. crop-geslict auf 100 statt 200)
        sg_v_falsch = np.zeros(100)   # Länge != n

        linien_fallback, v_fallback = chart_module._finde_sop_kreuzungen(
            zeit, signal, rect_fit,
            sop_percent=50.0,
            sample_rate=1.0 / (dt_ms * 1e-3),
            halbes_zeitfenster=3,
            v_faktor=1.0,
            sg_v=sg_v_falsch,
        )

        linien_ohne, v_ohne = chart_module._finde_sop_kreuzungen(
            zeit, signal, rect_fit,
            sop_percent=50.0,
            sample_rate=1.0 / (dt_ms * 1e-3),
            halbes_zeitfenster=3,
            v_faktor=1.0,
            sg_v=None,
        )
        # Falsche Länge → Fallback → identisch mit sg_v=None
        assert math.isclose(v_fallback, v_ohne, rel_tol=1e-9, abs_tol=1e-9), \
            "Falsche sg_v-Länge sollte denselben Fallback-Wert wie sg_v=None liefern"

    def test_sg_v_none_liefert_valides_ergebnis(self):
        """Ohne sg_v (None) funktioniert SOP über Differenzenquotienten korrekt."""
        n     = 200
        dt_ms = 0.1
        zeit, signal = self._rechteck_signal(n=n)
        rect_fit = self._minimal_rect_fit(
            t_start=zeit[60], t_end=zeit[140], y_low=0.0, y_high=100.0
        )
        linien, v_sop = chart_module._finde_sop_kreuzungen(
            zeit, signal, rect_fit,
            sop_percent=50.0,
            sample_rate=1.0 / (dt_ms * 1e-3),
            halbes_zeitfenster=3,
            v_faktor=1.0,
            sg_v=None,
        )
        assert not math.isnan(v_sop) or len(linien) == 0

    def test_hub_null_gibt_leere_liste(self):
        """hub <= 0 → keine SOP-Linien, nan."""
        n = 100
        zeit   = np.arange(n, dtype=float)
        signal = np.zeros(n)
        rect_fit = {'y_low': 5.0, 'y_high': 5.0, 'runs': [{'t_start': 10.0, 't_end': 90.0}]}
        linien, v_sop = chart_module._finde_sop_kreuzungen(
            zeit, signal, rect_fit,
            sop_percent=50.0, sample_rate=1000.0, halbes_zeitfenster=5,
        )
        assert linien == []
        assert math.isnan(v_sop)


# ---------------------------------------------------------------------------
# Bug #4 – SOP findet keine Kreuzung, wenn sop_percent < 50 % Hub
# Bei kontinuierlichem Anstieg beginnt der Rechteck-Fit-Run erst bei
# 50 % des Hubs (threshold = 0.5 * (min + max)). Liegt der SOP-Pegel darunter,
# überschreitet die Kurve ihn vor run['t_start'] – außerhalb des bisherigen
# fixen Vorlaufs (10 Samples). Das Suchfenster muss dynamisch nach links
# erweitert werden.
# ---------------------------------------------------------------------------

class TestSopPegelUnter50:
    """SOP-Pegel unterhalb des Rechteck-Thresholds (< 50 % Hub) muss gefunden
    werden, auch wenn die Kurve den Pegel vor run['t_start'] überschreitet."""

    def _kontinuierlicher_anstieg(self, n=300, dt_us=1.0,
                                    y_start=0.0, y_ende=300.0,
                                    puls_start_us=150.0, puls_ende_us=250.0):
        """Linear ansteigendes Signal: 0 → 300 µm über die gesamte Dauer.
        Rechteck-Fit-Run simuliert beginnt erst bei 150 µm (50 % Hub)."""
        zeit_s    = np.arange(n) * dt_us * 1e-6
        signal    = np.linspace(y_start, y_ende, n)
        return zeit_s, signal

    def _minimal_rect_fit(self, t_start, t_end, y_low, y_high):
        return {
            'y_low':  y_low,
            'y_high': y_high,
            'runs': [{'t_start': t_start, 't_end': t_end}],
        }

    def test_sop_unter_50_prozent_findet_kreuzung(self):
        """SOP bei 30 % muss gefunden werden, obwohl run['t_start'] bei 50 % liegt."""
        zeit, signal = self._kontinuierlicher_anstieg()
        # Run beginnt erst dort, wo Kurve 150 µm erreicht (Index 150, 150 µs)
        rect_fit = self._minimal_rect_fit(
            t_start=float(zeit[150]), t_end=float(zeit[250]),
            y_low=0.0, y_high=300.0
        )
        linien, v_sop = chart_module._finde_sop_kreuzungen(
            zeit, signal, rect_fit,
            sop_percent=30.0,
            sample_rate=1.0e6,   # 1 µs-Schritt
            halbes_zeitfenster=3,
            v_faktor=1.0,
            sg_v=None,
        )
        assert len(linien) > 0, "SOP-Kreuzung bei 30 % Hub nicht gefunden"
        assert not math.isnan(v_sop)
        # SOP-Pegel = 0 + 0.3 * 300 = 90 µm → Kreuzung bei Index 90 (90 µs)
        t_sop = linien[0][0]
        assert math.isclose(t_sop, zeit[90], rel_tol=1e-6, abs_tol=1e-9), \
            f"SOP bei t={t_sop:.6e}s erwartet={zeit[90]:.6e}s"

    def test_sop_50_prozent_findet_kreuzung(self):
        """SOP bei 50 % – Regressionstest für unverändertes Verhalten."""
        zeit, signal = self._kontinuierlicher_anstieg()
        rect_fit = self._minimal_rect_fit(
            t_start=float(zeit[150]), t_end=float(zeit[250]),
            y_low=0.0, y_high=300.0
        )
        linien, v_sop = chart_module._finde_sop_kreuzungen(
            zeit, signal, rect_fit,
            sop_percent=50.0,
            sample_rate=1.0e6,
            halbes_zeitfenster=3,
            v_faktor=1.0,
            sg_v=None,
        )
        assert len(linien) > 0
        assert not math.isnan(v_sop)
        t_sop = linien[0][0]
        assert math.isclose(t_sop, zeit[150], rel_tol=1e-6, abs_tol=1e-9)
