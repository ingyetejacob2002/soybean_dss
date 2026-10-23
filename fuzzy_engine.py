"""Mamdani fuzzy inference engine (Section 3.5) built with scikit-fuzzy.
Input: symptom severity 0..1 (triangular sets Low/Moderate/High). Output: disease score 0..100 (centroid)."""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

X = np.arange(0, 1.001, 0.01)
Y = np.arange(0, 100.5, 1)
IN_MF = {"Low": [0, 0, .4], "Moderate": [.3, .5, .7], "High": [.6, 1, 1]}
OUT_MF = {"Low": [0, 0, 40], "Medium": [30, 50, 70], "High": [60, 100, 100]}


def memberships(x):
    """Fuzzification of one crisp severity value."""
    return {k: float(fuzz.interp_membership(X, fuzz.trimf(X, p), x)) for k, p in IN_MF.items()}


def _expected(ant, level, strict=True):
    """Fuzzy term for the severity a disease expects. Strict: High or Moderate only; lenient: Moderate also accepts High."""
    if level == "High":
        return ant["High"]
    return ant["Moderate"] if strict else (ant["Moderate"] | ant["High"])


class FuzzyDiagnoser:
    def __init__(self, profiles, symptom_keys):
        """profiles: {disease_id: [(symptom_key, 'High'|'Moderate'), ...]} ordered by importance."""
        self.profiles = profiles
        self.ants = {}
        for k in symptom_keys:
            a = ctrl.Antecedent(X, k)
            for n, p in IN_MF.items():
                a[n] = fuzz.trimf(X, p)
            self.ants[k] = a
        self.systems = {d: self._build(d, p) for d, p in profiles.items()}
        self._urgency = self._build_urgency()

    def _build(self, did, prof):
        out = ctrl.Consequent(Y, "score_" + did, defuzzify_method="centroid")
        for n, p in OUT_MF.items():
            out[n] = fuzz.trimf(Y, p)
        A = self.ants
        k = [(A[s], l) for s, l in prof]
        high = _expected(*k[0]) & _expected(*k[1]) & _expected(*k[2]) if len(k) > 2 else _expected(*k[0]) & _expected(*k[1])
        rules = [ctrl.Rule(high, out["High"]),
                 ctrl.Rule(~k[0][0]["Low"] & ~k[1][0]["Low"] & (~k[2][0]["Low"] if len(k) > 2 else ~k[1][0]["Low"]), out["Medium"]),
                 ctrl.Rule(k[0][0]["Low"] | k[1][0]["Low"] | (k[2][0]["Low"] if len(k) > 2 else k[1][0]["Low"]), out["Low"])]
        if len(k) > 3:  # supporting symptoms raise a strong partial match to High
            support = _expected(*k[3], False) if len(k) == 4 else _expected(*k[3], False) | _expected(*k[4], False)
            rules.append(ctrl.Rule(_expected(*k[0], False) & _expected(*k[1], False) & support, out["High"]))
        return ctrl.ControlSystem(rules), out.label

    def _build_urgency(self):
        score = ctrl.Antecedent(Y, "score"); prog = ctrl.Antecedent(X, "progression")
        urg = ctrl.Consequent(Y, "urgency", defuzzify_method="centroid")
        for n, p in OUT_MF.items(): score[n] = fuzz.trimf(Y, p)
        for n, p in IN_MF.items(): prog[n] = fuzz.trimf(X, p)
        urg["Monitor"] = fuzz.trimf(Y, [0, 0, 40]); urg["Act soon"] = fuzz.trimf(Y, [30, 50, 70]); urg["Act now"] = fuzz.trimf(Y, [60, 100, 100])
        r = [ctrl.Rule(score["High"] & (prog["Moderate"] | prog["High"]), urg["Act now"]),
             ctrl.Rule(score["High"] & prog["Low"], urg["Act soon"]),
             ctrl.Rule(score["Medium"] & prog["High"], urg["Act soon"]),
             ctrl.Rule(score["Medium"] & (prog["Low"] | prog["Moderate"]), urg["Monitor"]),
             ctrl.Rule(score["Low"], urg["Monitor"])]
        return ctrl.ControlSystem(r)

    def diagnose(self, values, progression=0.0):
        """values: {symptom_key: severity 0..1}. Returns ranked scores, urgency and an explanation."""
        scores = {}
        for did, (system, outname) in self.systems.items():
            sim = ctrl.ControlSystemSimulation(system)
            for s, _ in self.profiles[did]:
                sim.input[s] = float(values.get(s, 0))
            try:
                sim.compute(); scores[did] = round(float(sim.output[outname]), 1)
            except Exception:
                scores[did] = 0.0
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        usim = ctrl.ControlSystemSimulation(self._urgency)
        usim.input["score"] = ranked[0][1]; usim.input["progression"] = float(progression)
        usim.compute()
        u = float(usim.output["urgency"])
        return ranked, round(u, 1), ("Act now" if u >= 62 else "Act soon" if u >= 38 else "Monitor")

    def explain(self, did, values):
        """Membership degree of each key symptom in the expected set (why the rule fired)."""
        rows = []
        for s, level in self.profiles[did][:3]:
            m = memberships(float(values.get(s, 0)))
            deg = m["High"] if level == "High" else max(m["Moderate"], m["High"])
            rows.append((s, level, round(deg, 2)))
        return rows
