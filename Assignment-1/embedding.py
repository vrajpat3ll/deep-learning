import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_glove_embeddings(path, max_vocab=None, embedding_dim=None):
    """
    Load GloVe/word2vec text file -> dict(word -> np.array).
    If embedding_dim is provided, lines that don't match are ignored.
    max_vocab: optional cap on number of lines to read.
    """
    emb = {}
    with open(path, 'r', encoding='utf8', errors='ignore') as f:
        for i, line in enumerate(f):
            try:
                if max_vocab and i >= max_vocab:
                    break
                parts = line.rstrip().split()
                if len(parts) < 2:
                    continue
                word = parts[0]
                # print(f"{i}) Loading {word}...")
                vec = np.asarray(parts[1:], dtype=np.float32)
                if embedding_dim and vec.shape[0] != embedding_dim:
                    continue
                emb[word] = vec
            except Exception as e:
                continue
                # print(f"[GLoVE] Could not load {i}th word: {parts[:4]}")
    if len(emb) == 0:
        raise RuntimeError("No embeddings loaded. Check path/format.")
    # infer dim
    dim = next(iter(emb.values())).shape[0]
    return emb, dim


def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    return text.split()


def texts_to_avg_embeddings(texts, emb_index, emb_dim, normalize=True):
    """
    texts: list[str]; emb_index: dict(word->vec); emb_dim: int
    Returns numpy array shape (N, emb_dim)
    OOV tokens skipped; if no known token found, returns zero vector.
    """
    X = np.zeros((len(texts), emb_dim), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = simple_tokenize(t)
        vecs = [emb_index[w] for w in toks if w in emb_index]
        if not vecs:
            continue
        arr = np.stack(vecs, axis=0)
        mean = arr.mean(axis=0)
        if normalize:
            norm = np.linalg.norm(mean)
            if norm > 0:
                mean = mean / norm
        X[i] = mean
    return X


def texts_to_tfidf_weighted_embeddings(X_train_texts, texts_to_transform, emb_index, emb_dim,
                                       tfidf_vectorizer=None, norm=True):
    """
    Compute TF-IDF weights fitted on X_train_texts (if tfidf_vectorizer is None).
    Returns embeddings for texts_to_transform (can be X_val or X_test).
    tfidf_vectorizer: optional TfidfVectorizer already fitted on X_train_texts.
    """
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(
            1, 2), stop_words='english', max_features=30000)
        tfidf_vectorizer.fit(X_train_texts)
    # get feature names (vocab of ngrams). But we need token-level weights, so easier approach:
    # We'll compute token-level TFIDF by using analyzer='word' and access token index.
    # If the vectorizer uses ngrams, we still can use token weights but they won't correspond exactly.
    # For simplicity, ensure tfidf_vectorizer uses analyzer='word' and appropriate tokenization for your texts.
    # Build idf map:
    idf = {}
    if hasattr(tfidf_vectorizer, 'vocabulary_'):
        inv_vocab = {idx: token for token,
                     idx in tfidf_vectorizer.vocabulary_.items()}
        idf_vals = tfidf_vectorizer.idf_
        for token, idx in tfidf_vectorizer.vocabulary_.items():
            idf[token] = idf_vals[idx]
    # fallback: equal weights if something missing
    default_idf = 1.0

    X = np.zeros((len(texts_to_transform), emb_dim), dtype=np.float32)
    for i, text in enumerate(texts_to_transform):
        toks = simple_tokenize(text)
        weighted = []
        weights = []
        for tok in toks:
            if tok in emb_index:
                w = idf.get(tok, default_idf)
                weighted.append(emb_index[tok] * w)
                weights.append(w)
        if not weighted:
            continue
        weighted = np.stack(weighted, axis=0)
        weights = np.asarray(weights, dtype=np.float32)
        vec = weighted.sum(axis=0) / (weights.sum() + 1e-12)
        if norm:
            nrm = np.linalg.norm(vec)
            if nrm > 0:
                vec = vec / nrm
        X[i] = vec
    return X, tfidf_vectorizer

# -------------------------
# SIF postprocessing: remove 1st principal component
# -------------------------


def remove_pc(X, npc=1):
    """
    Smooth Inverse Frequency (SIF) style postprocess: remove first principal component.
    X: (N, D)
    """
    if X.shape[0] <= 1 or npc <= 0:
        return X
    # mean-center
    Xc = X - X.mean(axis=0, keepdims=True)
    # compute top principal vector via SVD
    try:
        u, s, vt = np.linalg.svd(Xc, full_matrices=False)
        pc = vt[:npc]  # (npc, D)
        X_proj = Xc - Xc.dot(pc.T).dot(pc)
        return X_proj
    except Exception:
        return X
