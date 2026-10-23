"""Soybean Disease Decision Support System: Flask + SQLite + scikit-fuzzy (Chapter 3.7)."""
import os, sqlite3, json
from flask import Flask, render_template, request, g, abort
from knowledge import SYMPTOMS, GROUPS, DISEASES
from fuzzy_engine import FuzzyDiagnoser, IN_MF, OUT_MF

app = Flask(__name__)
DB = os.path.join(app.root_path, "instance", "soybean.db")
SYM = {s[0]: s for s in SYMPTOMS}

def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB); g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d: d.close()

def init_db():
    c = sqlite3.connect(DB)
    c.executescript("""
    CREATE TABLE IF NOT EXISTS disease(Disease_ID TEXT PRIMARY KEY, Disease_Name TEXT, Description TEXT, Plant_Part TEXT, Cause TEXT, Image TEXT);
    CREATE TABLE IF NOT EXISTS symptom(Symptom_ID TEXT PRIMARY KEY, Symptom_Name TEXT, Description TEXT, Grp TEXT);
    CREATE TABLE IF NOT EXISTS disease_symptom(Disease_ID TEXT, Symptom_ID TEXT, Severity TEXT, Rank INTEGER);
    CREATE TABLE IF NOT EXISTS treatment(Treatment_ID INTEGER PRIMARY KEY AUTOINCREMENT, Disease_ID TEXT, Treatment TEXT, Management TEXT);
    CREATE TABLE IF NOT EXISTS diagnosis_log(id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT DEFAULT CURRENT_TIMESTAMP, top_disease TEXT, score REAL, urgency TEXT, inputs TEXT);
    """)
    if not c.execute("SELECT 1 FROM disease").fetchone():
        for s in SYMPTOMS: c.execute("INSERT INTO symptom VALUES(?,?,?,?)", s)
        for d in DISEASES:
            c.execute("INSERT INTO disease VALUES(?,?,?,?,?,?)", (d["id"], d["name"], d["desc"], d["part"], d["cause"], d["image"]))
            for i, (s, lvl) in enumerate(d["profile"]): c.execute("INSERT INTO disease_symptom VALUES(?,?,?,?)", (d["id"], s, lvl, i))
            for t in d["treat"]: c.execute("INSERT INTO treatment(Disease_ID,Treatment,Management) VALUES(?,?,NULL)", (d["id"], t))
            for m in d["manage"]: c.execute("INSERT INTO treatment(Disease_ID,Treatment,Management) VALUES(?,NULL,?)", (d["id"], m))
        c.commit()
    c.close()

def load_engine():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    prof = {}
    for r in c.execute("SELECT * FROM disease_symptom ORDER BY Disease_ID, Rank"):
        prof.setdefault(r["Disease_ID"], []).append((r["Symptom_ID"], r["Severity"]))
    keys = [r["Symptom_ID"] for r in c.execute("SELECT Symptom_ID FROM symptom")]
    c.close()
    return FuzzyDiagnoser(prof, keys)

os.makedirs(os.path.dirname(DB), exist_ok=True)
init_db(); ENGINE = load_engine()

def disease_row(did):
    d = db().execute("SELECT * FROM disease WHERE Disease_ID=?", (did,)).fetchone()
    if not d: abort(404)
    t = db().execute("SELECT Treatment, Management FROM treatment WHERE Disease_ID=?", (did,)).fetchall()
    sy = db().execute("SELECT Symptom_ID, Severity FROM disease_symptom WHERE Disease_ID=? ORDER BY Rank", (did,)).fetchall()
    return dict(d), [r["Treatment"] for r in t if r["Treatment"]], [r["Management"] for r in t if r["Management"]], [(SYM[r["Symptom_ID"]][1], r["Severity"]) for r in sy]

@app.context_processor
def helpers():
    return dict(level=lambda v: "None" if v < .05 else "Low" if v < .35 else "Moderate" if v < .65 else "High")

@app.route("/")
def home():
    n = db().execute("SELECT COUNT(*) FROM disease").fetchone()[0]
    return render_template("home.html", n=n, nsym=len(SYMPTOMS))

@app.route("/diagnose", methods=["GET", "POST"])
def diagnose():
    if request.method == "GET":
        return render_template("diagnose.html", symptoms=SYMPTOMS, groups=GROUPS, mf=IN_MF)
    vals = {k: min(max(int(request.form.get(k, 0) or 0), 0), 10) / 10 for k in SYM}
    prog = min(max(int(request.form.get("progression", 0) or 0), 0), 10) / 10
    ranked, urgency_val, urgency = ENGINE.diagnose(vals, prog)
    top, score = ranked[0]
    if score < 40 or not any(vals.values()):
        return render_template("result.html", insufficient=True, ranked=[], names={}, vals=vals)
    d, treat, manage, _ = disease_row(top)
    names = {r["Disease_ID"]: r["Disease_Name"] for r in db().execute("SELECT Disease_ID, Disease_Name FROM disease")}
    why = [(SYM[s][1], lvl, deg, vals.get(s, 0)) for s, lvl, deg in ENGINE.explain(top, vals)]
    db().execute("INSERT INTO diagnosis_log(top_disease,score,urgency,inputs) VALUES(?,?,?,?)", (top, score, urgency, json.dumps(vals))); db().commit()
    return render_template("result.html", insufficient=False, d=d, treat=treat, manage=manage, score=score, urgency=urgency,
                           ranked=[(did, names[did], sc) for did, sc in ranked[:5]], why=why, vals=vals, prog=prog)

@app.route("/library")
def library():
    part = request.args.get("part")
    rows = db().execute("SELECT * FROM disease" + (" WHERE Plant_Part=?" if part else ""), (part,) if part else ()).fetchall()
    return render_template("library.html", rows=rows, groups=["Leaves", "Stem and pods", "Seeds", "Whole plant and roots"], part=part)

@app.route("/library/<did>")
def disease(did):
    d, treat, manage, syms = disease_row(did)
    return render_template("disease.html", d=d, treat=treat, manage=manage, syms=syms)

@app.route("/how-it-works")
def about():
    return render_template("about.html", in_mf=IN_MF, out_mf=OUT_MF)

if __name__ == "__main__":
    app.run(debug=True)
