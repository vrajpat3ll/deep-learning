"""
- For each classical ML model:
    * Run K-fold CV on TRAIN: fit TF-IDF inside each fold (no leakage)
    * Record per-fold TRAIN & VAL metrics (accuracy, precision_macro, recall_macro, f1_macro)
    * Aggregate mean+/-std for TRAIN and VAL
    * Fit TF-IDF on full TRAIN, train final model on full TRAIN, evaluate on TEST
    * Save results / confusion matrix / classification report / top features (if available)
- Models: LogisticRegression, LinearSVC, SGDClassifier, MultinomialNB, RandomForest, ExtraTrees

Run:
    python main.py --input Full_data.csv --out_dir ./results
"""

import os
import argparse
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import utils
from run_model import run_model_kfold


def make_text_column(df):
    s = df["Statement"].fillna("").astype(str)
    j = df["Justification"].fillna("").astype(str)
    text = (s + " " + j).str.strip().apply(utils.clean_text)
    return text


def map_binary(lbl):
    """Binary mapping for LIAR dataset labels."""
    true_set = {"half_true", "mostly_true", "true"}
    false_set = {"mostly_false", "false", "pants_fire"}
    ll = str(lbl).strip().lower()
    if ll in true_set:
        return 1
    elif ll in false_set:
        return 0
    else:
        return float('inf')


def main(args):
    start_time = time.time()
    utils.ensure_dir(args.out_dir)

    # load data
    df = pd.read_csv(args.input)
    df["label_binary"] = df["label"].apply(map_binary)

    required_cols = ["Statement", "Justification", "label"]
    for c in required_cols:
        if c not in df.columns:
            raise RuntimeError(
                f"Required column '{c}' not found in input CSV.")

    df["text"] = make_text_column(df)
    texts = df["text"].values
    labels = df["label"].astype(str).str.strip().values
    labels_binary = df["label_binary"].astype(float).values

    # train/test split (stratified)
    mX_train, mX_test, my_train, my_test = train_test_split(texts, labels, test_size=args.test_size,
                                                            stratify=labels, random_state=args.random_state)
    bX_train, bX_test, by_train, by_test = train_test_split(texts, labels_binary, test_size=args.test_size,
                                                            stratify=labels_binary, random_state=args.random_state)

    print(f"Train/test sizes: {len(mX_train)} / {len(mX_test)}")
    print(f"Using {args.feature_type} embeddings/vectors...")

    emb_index, emb_dim = None, None
    if args.feature_type in ['emb_avg', 'emb_tfidf']:
        try:
            from embedding import load_glove_embeddings
            print(f"Loading GloVe embeddings from: {args.glove_path}")
            emb_index, emb_dim = load_glove_embeddings(
                args.glove_path, embedding_dim=args.emb_dim)
            print(f"Loaded {len(emb_index)} embeddings, dim={emb_dim}")
        except Exception as e:
            print(
                f"Warning: Could not load embeddings ({e}), falling back to TF-IDF")
            args.feature_type = 'tfidf'
            emb_index, emb_dim = None, None

    # define models (instances)
    models = [
        ("logreg", LogisticRegression(max_iter=2000, class_weight='balanced',
         solver='saga', random_state=args.random_state)),
        ("linear_svc", LinearSVC(class_weight='balanced',
         max_iter=10000, random_state=args.random_state)),
        ("sgd_hinge", SGDClassifier(loss='hinge', max_iter=5000,
         tol=1e-4, class_weight='balanced', random_state=args.random_state)),
        ("multinb", MultinomialNB()),
        ("random_forest", RandomForestClassifier(n_estimators=200,
         class_weight='balanced', n_jobs=args.n_jobs, random_state=args.random_state)),
        ("extra_trees", ExtraTreesClassifier(n_estimators=200,
         class_weight='balanced', n_jobs=args.n_jobs, random_state=args.random_state))
    ]

    print("\n" + "="*80)
    print("RUNNING MULTICLASS CLASSIFICATION (6 labels)")
    print("="*80)

    # run each model
    summary = {}
    # run for multiclass classification
    for name, inst in models:
        print("\n" + "="*60)
        print("Running model (Multiclass):", name)
        res = run_model_kfold(name, inst, mX_train, my_train, mX_test, my_test,
                              emb_dim=emb_dim, emb_index=emb_index,
                              feature_type=args.feature_type, k=args.k,
                              max_features=args.max_features,
                              ngram_range=tuple(
                                  map(int, args.ngram.split(","))),
                              out_dir=args.out_dir, random_state=args.random_state)
        summary[f"multi_{name}"] = res

    print("\n" + "="*80)
    print("RUNNING BINARY CLASSIFICATION (2 labels)")
    print("="*80)

    # run for binary classification
    for name, inst in models:
        print("\n" + "="*60)
        print("Running model (Binary):", name)
        res = run_model_kfold(name, inst, bX_train, by_train, bX_test, by_test,
                              emb_dim=emb_dim, emb_index=emb_index,
                              feature_type=args.feature_type, k=args.k,
                              max_features=args.max_features,
                              ngram_range=tuple(
                                  map(int, args.ngram.split(","))),
                              out_dir=args.out_dir, random_state=args.random_state)
        summary[f"bin_{name}"] = res

    # save summary
    utils.save_json(summary, os.path.join(
        args.out_dir, "all_models_summary.json"))

    elapsed = time.time() - start_time
    print("\nAll models finished. Results saved to:", args.out_dir)
    print("Elapsed time: {:.1f}s".format(elapsed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run classical ML models with K-fold CV on TRAIN and final TEST evaluation.")
    parser.add_argument("-i", "--input", type=str,
                        default="Full_data.csv", help="Input CSV path")
    parser.add_argument("-o", "--out_dir", type=str,
                        default="./results_ml", help="Output directory")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("-k", type=int, default=5,
                        help="K for K-fold CV on TRAIN")
    parser.add_argument("--max_features", type=int,
                        default=3000, help="max TF-IDF features")
    parser.add_argument("--ngram", type=str, default="1,2",
                        help="ngram range as 'min,max' e.g. '1,2'")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("-j", "--n_jobs", type=int, default=2)
    parser.add_argument("-ft", "--feature_type", type=str, default='tfidf',
                        choices=['tfidf', 'emb_avg', 'emb_tfidf'],
                        help="Feature extraction method")
    parser.add_argument("--emb_dim", type=int, default=50,
                        help="Embedding dimension")
    parser.add_argument("--glove_path", type=str, default=r"./wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt",
                        help="Path to GloVe embeddings file")

    args = parser.parse_args()
    main(args)
