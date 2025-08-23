Twitter Sentiment Full Project (Reference-Style, Tri-class)
=========================================================

This project keeps your original single-file structure and charts,
but adapts it to **Saurabh's Twitter Sentiment dataset** (labels: -1, 0, 1).

Files
-----
- train_full_reference_style.py   <-- Run this (all-in-one EDA + training + saving)
- data/sample_saurabh.csv         <-- tiny fallback sample
- models/                         <-- artifacts saved here
- requirements.txt

How to use
----------
1) Download Saurabh's Kaggle dataset CSV and place it at:
   data/saurabh_twitter_sentiment.csv

2) Create a virtual environment (recommended) and install deps:
   python -m venv .venv
   .venv\Scripts\activate    (Windows)  OR  source .venv/bin/activate
   pip install -r requirements.txt

3) Run the script:
   python train_full_reference_style.py

4) The script will:
   - Perform EDA (plots)
   - Train multiple models (Naive Bayes, LogisticRegression, SVC, trees, ensembles, Stacking)
   - Compare accuracy/precision
   - Save final vectorizer and model to models/vectorizer.pkl and models/model.pkl

Notes
-----
- If you haven't added the Kaggle CSV yet, the script will use the small sample to run.
- The script preserves your original printouts and sections; only safe guards and tri-class support were added.
