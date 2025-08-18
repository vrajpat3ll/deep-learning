import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import clone
from embedding import texts_to_avg_embeddings, texts_to_tfidf_weighted_embeddings
import utils


def run_model_kfold(model_name, model_instance,
                    texts_train, y_train, texts_test, y_test,
                    feature_type="tfidf",   # "tfidf", "emb_avg", "emb_tfidf"
                    emb_index=None, emb_dim=None,
                    k=5, max_features=3000, ngram_range=(1,2),
                    out_dir="./results", random_state=42):
    """
    runs K-fold CV + final test eval with either TF-IDF or embeddings
    """

    out_dir = os.path.join(out_dir, model_name + "_" + feature_type)
    utils.ensure_dir(out_dir)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    train_fold_metrics, val_fold_metrics = [], []

    print(f"[{model_name}] Running {k}-fold CV (feature_type={feature_type})...")

    fold = 0
    for train_idx, val_idx in skf.split(texts_train, y_train):
        fold += 1
        X_tr_texts = [texts_train[i] for i in train_idx]
        y_tr = [y_train[i] for i in train_idx]
        X_val_texts = [texts_train[i] for i in val_idx]
        y_val = [y_train[i] for i in val_idx]


        # -------- feature extraction --------
        if feature_type == "tfidf":
            vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words="english")
            X_tr = vec.fit_transform(X_tr_texts)
            X_val = vec.transform(X_val_texts)

        elif feature_type == "emb_avg":
            if emb_index is None: raise ValueError("Need emb_index for emb_avg")
            X_tr = texts_to_avg_embeddings(X_tr_texts, emb_index, emb_dim)
            X_val = texts_to_avg_embeddings(X_val_texts, emb_index, emb_dim)

        elif feature_type == "emb_tfidf":
            if emb_index is None: raise ValueError("Need emb_index for emb_tfidf")
            tfidf = TfidfVectorizer(analyzer='word', stop_words='english', max_features=max_features)
            tfidf.fit(X_tr_texts)
            X_tr, _ = texts_to_tfidf_weighted_embeddings(X_tr_texts, X_tr_texts, emb_index, emb_dim, tfidf_vectorizer=tfidf)
            X_val, _ = texts_to_tfidf_weighted_embeddings(X_tr_texts, X_val_texts, emb_index, emb_dim, tfidf_vectorizer=tfidf)

        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        if model_name == "multinb" and feature_type in ['emb_avg', 'emb_tfidf']:
            X_tr = np.maximum(X_tr, 0)  # clip negative values to 0
            X_val = np.maximum(X_val, 0)
            

        # -------- train + evaluate --------
        clf = clone(model_instance)
        clf.fit(X_tr, y_tr)
        y_tr_pred = clf.predict(X_tr)
        y_val_pred = clf.predict(X_val)

        train_fold_metrics.append(utils.compute_metrics(y_tr, y_tr_pred))
        val_fold_metrics.append(utils.compute_metrics(y_val, y_val_pred))
        print(f" > fold {fold}/{k} val_f1={val_fold_metrics[-1]['f1_macro']:.4f}")

    # aggregate
    train_df = pd.DataFrame(train_fold_metrics)
    val_df = pd.DataFrame(val_fold_metrics)
    results = {
        "train_mean": train_df.mean().to_dict(),
        "train_std":  train_df.std().to_dict(),
        "val_mean":   val_df.mean().to_dict(),
        "val_std":    val_df.std().to_dict()
    }

    # -------- final train on full data --------
    print(f"[{model_name}] Training final model on full TRAIN and evaluating on TEST...")
    
    if feature_type == "tfidf":
        vec_full = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words="english")
        X_train_full = vec_full.fit_transform(texts_train)
        X_test_full = vec_full.transform(texts_test)

    elif feature_type == "emb_avg":
        X_train_full = texts_to_avg_embeddings(texts_train, emb_index, emb_dim)
        X_test_full = texts_to_avg_embeddings(texts_test, emb_index, emb_dim)

    elif feature_type == "emb_tfidf":
        tfidf_full = TfidfVectorizer(analyzer='word', stop_words='english', max_features=30000)
        tfidf_full.fit(texts_train)
        X_train_full, _ = texts_to_tfidf_weighted_embeddings(texts_train, texts_train, emb_index, emb_dim, tfidf_vectorizer=tfidf_full)
        X_test_full, _  = texts_to_tfidf_weighted_embeddings(texts_train, texts_test, emb_index, emb_dim, tfidf_vectorizer=tfidf_full)

    if model_name == "multinb" and feature_type in ['emb_avg', 'emb_tfidf']:
        X_train_full = np.maximum(X_train_full, 0)  # clip negative values to 0
        X_test_full = np.maximum(X_test_full, 0)


    clf_final = clone(model_instance)
    clf_final.fit(X_train_full, y_train)
    y_test_pred = clf_final.predict(X_test_full)

    results["test_metrics"] = utils.compute_metrics(y_test, y_test_pred)
    results["test_classification_report"] = classification_report(y_test, y_test_pred, output_dict=True)
    results["test_confusion_matrix"] = confusion_matrix(y_test, y_test_pred).tolist()
    results["classes"] = list(getattr(clf_final, "classes_", sorted(set(y_train))))

    # save
    pref = "bin" if len(set(y_train)) == 2 else "multi"
    utils.save_json(results, os.path.join(out_dir, f"{pref}_{model_name}_{feature_type}_results.json"))
    pd.DataFrame(train_fold_metrics).to_csv(os.path.join(out_dir, f"{pref}_train_fold_metrics_{feature_type}.csv"), index=False)
    pd.DataFrame(val_fold_metrics).to_csv(os.path.join(out_dir, f"{pref}_val_fold_metrics_{feature_type}.csv"), index=False)
    pd.DataFrame(results["test_classification_report"]).transpose().to_csv(os.path.join(out_dir, f"{pref}_test_classification_report_{feature_type}.csv"))
    pd.DataFrame(results["test_confusion_matrix"], index=results["classes"], columns=results["classes"]).to_csv(os.path.join(out_dir, f"{pref}_test_confusion_matrix_{feature_type}.csv"))

    print(f"[{model_name}] Done. test f1_macro={results['test_metrics']['f1_macro']:.4f} | acc={results['test_metrics']['accuracy']}")
    return results
