# SoyDoc: Soybean Disease Decision Support System (Fuzzy Logic)

Stack (per Chapter 3.2): Python, Flask, scikit-fuzzy (Mamdani, triangular sets, centroid), SQLite, HTML5/CSS3/JavaScript.

## Run
    pip install -r requirements.txt
    python app.py        # open http://127.0.0.1:5000

The SQLite database (instance/soybean.db) is created and seeded from knowledge.py on first run.

## Files
- fuzzy_engine.py : fuzzification, rule base, aggregation, centroid defuzzification, urgency system
- knowledge.py    : symptoms, diseases, symptom profiles, treatments (edit here, delete instance/soybean.db, restart)
- app.py          : Flask routes and database (disease, symptom, disease_symptom, treatment, diagnosis_log tables)
- templates/, static/ : interface and disease photos cropped from Figure 2.1 of the project document

Note: rules and treatments should be reviewed by a plant pathologist or supervisor (Section 3.9) before field use.
