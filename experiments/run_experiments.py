#!/usr/bin/env python3
"""Reproducible cross-corpus writing-assessment experiments.

The script trains CEFR classifiers on W&I, transfers their expected proficiency
scores to FCE, evaluates direct FCE scoring baselines, and audits out-of-fold
errors by first-language group. Raw corpora are not redistributed.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler


SEED = 20260831
ROOT = Path(__file__).resolve().parents[1]
WI_DIR = ROOT / "dataset_raw/extracted_wilocness/wi+locness/json"
FCE_DIR = ROOT / "dataset_raw/extracted_fce/fce/json"
OUT = ROOT / "results"
FIG = OUT / "figures"

TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")
SENT_RE = re.compile(r"[.!?]+")


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_wi() -> pd.DataFrame:
    records: list[dict] = []
    for split in ("train", "dev"):
        for broad in ("A", "B", "C"):
            for row in read_jsonl(WI_DIR / f"{broad}.{split}.json"):
                records.append(
                    {
                        "id": row["id"],
                        # A small number of anonymised records have no user ID.
                        # Treat each as a distinct writer rather than dropping text.
                        "userid": row.get("userid", f"missing-{row['id']}"),
                        "text": row["text"],
                        "cefr_fine": row["cefr"],
                        "label": {"A": 0, "B": 1, "C": 2}[broad],
                        "cefr": broad,
                        "split": split,
                    }
                )
    frame = pd.DataFrame(records)
    # The released train/dev files share writers. Repartition all labelled texts
    # at writer level so repeated submissions cannot inflate internal validity.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, dev_idx = next(splitter.split(frame, frame.label, groups=frame.userid))
    frame["eval_split"] = "train"
    frame.loc[dev_idx, "eval_split"] = "dev"
    overlap = set(frame.loc[frame.eval_split == "train", "userid"]) & set(
        frame.loc[frame.eval_split == "dev", "userid"]
    )
    if overlap:
        raise ValueError(f"W&I user leakage after grouped repartition: {len(overlap)} users")
    return frame


def load_fce() -> pd.DataFrame:
    rows: list[dict] = []
    for split in ("train", "dev", "test"):
        for row in read_jsonl(FCE_DIR / f"fce.{split}.json"):
            row = dict(row)
            row["split"] = split
            rows.append(row)
    responses = pd.DataFrame(rows)
    responses["script_score"] = pd.to_numeric(responses["script-s"], errors="coerce")
    if responses["script_score"].isna().any():
        raise ValueError("Non-numeric FCE script score encountered")

    scripts: list[dict] = []
    for script_id, part in responses.groupby("id", sort=False):
        if part["split"].nunique() != 1 or part["script_score"].nunique() != 1:
            raise ValueError(f"Inconsistent FCE script metadata for {script_id}")
        scripts.append(
            {
                "id": script_id,
                "text": "\n\n".join(part["text"].astype(str)),
                "script_score": float(part["script_score"].iloc[0]),
                "l1": str(part["l1"].iloc[0]),
                "age": str(part["age"].iloc[0]),
                "prompts": "+".join(sorted(part["q"].astype(str).unique())),
                "n_responses": int(len(part)),
                "split": str(part["split"].iloc[0]),
            }
        )
    frame = pd.DataFrame(scripts)
    if frame["id"].duplicated().any():
        raise ValueError("FCE script IDs are not unique after aggregation")
    return frame


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """Compute a compact bidirectional MTLD lexical-diversity estimate."""

    def one_direction(seq: list[str]) -> float:
        if not seq:
            return 0.0
        factors = 0.0
        types: set[str] = set()
        start = 0
        for index, token in enumerate(seq, start=1):
            types.add(token)
            length = index - start
            if length and len(types) / length <= threshold:
                factors += 1.0
                types = set()
                start = index
        remainder = len(seq) - start
        if remainder:
            ttr = len(types) / remainder
            factors += safe_div(1.0 - ttr, 1.0 - threshold)
        return safe_div(len(seq), factors) if factors else float(len(seq))

    return (one_direction(tokens) + one_direction(list(reversed(tokens)))) / 2.0


def surface_features(texts: list[str] | pd.Series) -> np.ndarray:
    rows: list[list[float]] = []
    for text in texts:
        tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(str(text))]
        counts = Counter(tokens)
        n_words = len(tokens)
        n_types = len(counts)
        n_sent = max(1, len(SENT_RE.findall(str(text))))
        n_chars = sum(len(token) for token in tokens)
        hapax = sum(value == 1 for value in counts.values())
        long_words = sum(len(token) >= 7 for token in tokens)
        paragraphs = max(1, len([p for p in re.split(r"\n\s*\n", str(text)) if p.strip()]))
        punctuation = sum(ch in ",;:!?" for ch in str(text))
        uppercase = sum(token[:1].isupper() for token in TOKEN_RE.findall(str(text)))
        rows.append(
            [
                math.log1p(n_words),
                math.log1p(n_sent),
                math.log1p(paragraphs),
                safe_div(n_words, n_sent),
                safe_div(n_chars, n_words),
                safe_div(n_types, math.sqrt(max(n_words, 1))),
                safe_div(hapax, n_words),
                safe_div(long_words, n_words),
                math.log1p(mtld(tokens)),
                safe_div(punctuation, n_words),
                safe_div(uppercase, n_words),
            ]
        )
    return np.asarray(rows, dtype=float)


def make_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=40_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=60_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def expected_level(probabilities: np.ndarray) -> np.ndarray:
    return probabilities @ np.arange(probabilities.shape[1], dtype=float)


def classification_ece(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            value += mask.mean() * abs((prediction[mask] == y[mask]).mean() - confidence[mask].mean())
    return float(value)


def wi_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    score = expected_level(probabilities)
    return {
        "QWK": cohen_kappa_score(y, prediction, weights="quadratic"),
        "Macro_F1": f1_score(y, prediction, average="macro"),
        "MAE_levels": mean_absolute_error(y, score),
        "Spearman": spearmanr(y, score).statistic,
        "ECE": classification_ece(y, probabilities),
    }


def fit_wi_models(wi: pd.DataFrame):
    train = wi[wi.eval_split == "train"].reset_index(drop=True)
    dev = wi[wi.eval_split == "dev"].reset_index(drop=True)
    y_train = train.label.to_numpy()
    y_dev = dev.label.to_numpy()

    scaler = StandardScaler()
    x_surface_train = scaler.fit_transform(surface_features(train.text))
    x_surface_dev = scaler.transform(surface_features(dev.text))
    vectorizer = make_vectorizer()
    x_text_train = vectorizer.fit_transform(train.text)
    x_text_dev = vectorizer.transform(dev.text)
    x_hybrid_train = sparse.hstack([x_text_train, sparse.csr_matrix(x_surface_train)], format="csr")
    x_hybrid_dev = sparse.hstack([x_text_dev, sparse.csr_matrix(x_surface_dev)], format="csr")

    specifications = {
        "Surface": (x_surface_train, x_surface_dev),
        "TF-IDF": (x_text_train, x_text_dev),
        "Hybrid": (x_hybrid_train, x_hybrid_dev),
    }
    result_rows = []
    fitted = {}
    for name, (x_train, x_dev) in specifications.items():
        model = LogisticRegression(
            C=1.0,
            max_iter=2500,
            class_weight="balanced",
            random_state=SEED,
            solver="lbfgs",
        )
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_dev)
        result_rows.append({"model": name, **wi_metrics(y_dev, probability)})
        fitted[name] = model

    bundle = {
        "scaler": scaler,
        "vectorizer": vectorizer,
        "model": fitted["Hybrid"],
    }
    return pd.DataFrame(result_rows), bundle


def wi_score(bundle: dict, texts: pd.Series | list[str]) -> np.ndarray:
    surface = bundle["scaler"].transform(surface_features(texts))
    text_matrix = bundle["vectorizer"].transform(texts)
    hybrid = sparse.hstack([text_matrix, sparse.csr_matrix(surface)], format="csr")
    return expected_level(bundle["model"].predict_proba(hybrid))


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    rounded = np.clip(np.rint(prediction), 0, 40).astype(int)
    return {
        "MAE": mean_absolute_error(y, prediction),
        "RMSE": math.sqrt(mean_squared_error(y, prediction)),
        "Pearson": pearsonr(y, prediction).statistic,
        "Spearman": spearmanr(y, prediction).statistic,
        "QWK": cohen_kappa_score(y.astype(int), rounded, weights="quadratic"),
    }


def regression_bootstrap_ci(
    y: np.ndarray, prediction: np.ndarray, iterations: int = 2000
) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    mae_values = []
    rho_values = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y), size=len(y))
        mae_values.append(mean_absolute_error(y[indices], prediction[indices]))
        rho_values.append(spearmanr(y[indices], prediction[indices]).statistic)
    return {
        "MAE_CI_low": float(np.nanquantile(mae_values, 0.025)),
        "MAE_CI_high": float(np.nanquantile(mae_values, 0.975)),
        "Spearman_CI_low": float(np.nanquantile(rho_values, 0.025)),
        "Spearman_CI_high": float(np.nanquantile(rho_values, 0.975)),
    }


def paired_mae_comparison(
    y: np.ndarray, first: np.ndarray, second: np.ndarray, iterations: int = 5000
) -> dict[str, float]:
    """Return first-minus-second MAE with a paired percentile interval."""
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y), size=len(y))
        deltas.append(
            mean_absolute_error(y[indices], first[indices])
            - mean_absolute_error(y[indices], second[indices])
        )
    return {
        "MAE_difference": mean_absolute_error(y, first) - mean_absolute_error(y, second),
        "CI_low": float(np.quantile(deltas, 0.025)),
        "CI_high": float(np.quantile(deltas, 0.975)),
    }


def prepare_fce_matrices(fce: pd.DataFrame):
    train = fce[fce.split == "train"].copy()
    dev = fce[fce.split == "dev"].copy()
    test = fce[fce.split == "test"].copy()
    train_dev = pd.concat([train, dev], ignore_index=True)

    scaler = StandardScaler()
    x_surface_train = scaler.fit_transform(surface_features(train.text))
    x_surface_dev = scaler.transform(surface_features(dev.text))
    vectorizer = make_vectorizer()
    x_text_train = vectorizer.fit_transform(train.text)
    x_text_dev = vectorizer.transform(dev.text)
    matrices = {
        "Surface": (x_surface_train, x_surface_dev),
        "TF-IDF": (x_text_train, x_text_dev),
        "Hybrid": (
            sparse.hstack([x_text_train, sparse.csr_matrix(x_surface_train)], format="csr"),
            sparse.hstack([x_text_dev, sparse.csr_matrix(x_surface_dev)], format="csr"),
        ),
    }
    return train, dev, test, train_dev, matrices


def select_ridge_alpha(train: pd.DataFrame, dev: pd.DataFrame, matrices: dict) -> dict[str, float]:
    selected = {}
    for name, (x_train, x_dev) in matrices.items():
        candidates = []
        for alpha in (1.0, 10.0, 100.0):
            model = Ridge(alpha=alpha, solver="lsqr")
            model.fit(x_train, train.script_score)
            candidates.append((mean_absolute_error(dev.script_score, model.predict(x_dev)), alpha))
        selected[name] = min(candidates)[1]
    return selected


def fit_fce_holdout(fce: pd.DataFrame, wi_bundle: dict):
    fce = fce.copy()
    fce["wi_score"] = wi_score(wi_bundle, fce.text)
    train, dev, test, train_dev, matrices = prepare_fce_matrices(fce)
    alphas = select_ridge_alpha(train, dev, matrices)

    scaler = StandardScaler()
    surf_train = scaler.fit_transform(surface_features(train_dev.text))
    surf_test = scaler.transform(surface_features(test.text))
    vectorizer = make_vectorizer()
    text_train = vectorizer.fit_transform(train_dev.text)
    text_test = vectorizer.transform(test.text)
    full_matrices = {
        "Surface": (surf_train, surf_test),
        "TF-IDF": (text_train, text_test),
        "Hybrid": (
            sparse.hstack([text_train, sparse.csr_matrix(surf_train)], format="csr"),
            sparse.hstack([text_test, sparse.csr_matrix(surf_test)], format="csr"),
        ),
    }

    rows = []
    predictions = {}
    transfer = LinearRegression().fit(train_dev[["wi_score"]], train_dev.script_score)
    transfer_pred = transfer.predict(test[["wi_score"]])
    y_test = test.script_score.to_numpy()
    rows.append({"model": "Transferred CEFR score", "alpha": np.nan, **regression_metrics(y_test, transfer_pred), **regression_bootstrap_ci(y_test, transfer_pred)})
    predictions["Transferred CEFR score"] = transfer_pred

    for name, (x_train, x_test) in full_matrices.items():
        model = Ridge(alpha=alphas[name], solver="lsqr")
        model.fit(x_train, train_dev.script_score)
        prediction = model.predict(x_test)
        rows.append({"model": name, "alpha": alphas[name], **regression_metrics(y_test, prediction), **regression_bootstrap_ci(y_test, prediction)})
        predictions[name] = prediction

    # Does the transferred proficiency signal add information beyond transparent features?
    z_scaler = StandardScaler()
    z_train = z_scaler.fit_transform(train_dev[["wi_score"]])
    z_test = z_scaler.transform(test[["wi_score"]])
    augmented_train = sparse.hstack([full_matrices["Hybrid"][0], sparse.csr_matrix(z_train)], format="csr")
    augmented_test = sparse.hstack([full_matrices["Hybrid"][1], sparse.csr_matrix(z_test)], format="csr")
    augmented = Ridge(alpha=alphas["Hybrid"], solver="lsqr").fit(augmented_train, train_dev.script_score)
    augmented_pred = augmented.predict(augmented_test)
    rows.append({"model": "Hybrid + transferred score", "alpha": alphas["Hybrid"], **regression_metrics(y_test, augmented_pred), **regression_bootstrap_ci(y_test, augmented_pred)})
    predictions["Hybrid + transferred score"] = augmented_pred

    test_predictions = test[["id", "script_score", "l1", "age", "prompts", "text", "wi_score"]].copy()
    for name, values in predictions.items():
        test_predictions[name] = values
    comparisons = []
    for first, second in (
        ("Transferred CEFR score", "TF-IDF"),
        ("Hybrid + transferred score", "Hybrid"),
        ("Hybrid + transferred score", "TF-IDF"),
    ):
        comparisons.append(
            {
                "first_model": first,
                "second_model": second,
                **paired_mae_comparison(y_test, predictions[first], predictions[second]),
            }
        )
    pd.DataFrame(comparisons).to_csv(OUT / "paired_model_comparisons.csv", index=False)
    return pd.DataFrame(rows), test_predictions, fce


def oof_tfidf_predictions(fce: pd.DataFrame, alpha: float = 1.0) -> pd.DataFrame:
    frame = fce.reset_index(drop=True).copy()
    bins = pd.qcut(frame.script_score, q=8, labels=False, duplicates="drop")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    prediction = np.full(len(frame), np.nan)
    for train_idx, test_idx in splitter.split(frame, bins):
        train = frame.iloc[train_idx]
        test = frame.iloc[test_idx]
        vectorizer = make_vectorizer()
        x_train = vectorizer.fit_transform(train.text)
        x_test = vectorizer.transform(test.text)
        model = Ridge(alpha=alpha, solver="lsqr").fit(x_train, train.script_score)
        prediction[test_idx] = model.predict(x_test)
    frame["prediction"] = prediction
    frame["signed_error"] = frame.prediction - frame.script_score
    frame["absolute_error"] = frame.signed_error.abs()
    frame["word_count"] = [len(TOKEN_RE.findall(text)) for text in frame.text]
    return frame


def bootstrap_mean_ci(values: np.ndarray, iterations: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    boot = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(iterations)])
    return tuple(np.quantile(boot, [0.025, 0.975]))


def subgroup_audit(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = []
    for l1, part in oof.groupby("l1"):
        if len(part) < 20:
            continue
        low, high = bootstrap_mean_ci(part.absolute_error.to_numpy())
        groups.append(
            {
                "L1": l1,
                "n": len(part),
                "MAE": part.absolute_error.mean(),
                "MAE_CI_low": low,
                "MAE_CI_high": high,
                "Mean_signed_error": part.signed_error.mean(),
                "Mean_human_score": part.script_score.mean(),
                "Mean_words": part.word_count.mean(),
            }
        )
    group_frame = pd.DataFrame(groups).sort_values("MAE", ascending=False)

    analysis = oof[oof.l1.isin(group_frame.L1)].copy()
    analysis["log_words"] = np.log1p(analysis.word_count)
    coefficient_rows = []
    fitted_models = {}
    joint_p = {}
    for outcome in ("signed_error", "absolute_error"):
        model = smf.ols(
            f"{outcome} ~ script_score + log_words + C(age) + C(l1)", data=analysis
        ).fit(cov_type="HC3")
        fitted_models[outcome] = model
        outcome_rows = []
        for term, value in model.params.items():
            if term.startswith("C(l1)"):
                outcome_rows.append(
                    {
                        "outcome": outcome,
                        "term": term,
                        "coefficient": value,
                        "robust_se": model.bse[term],
                        "p_value": model.pvalues[term],
                    }
                )
        raw_p = [row["p_value"] for row in outcome_rows]
        adjusted_p = multipletests(raw_p, method="fdr_bh")[1] if raw_p else []
        for row, p_value in zip(outcome_rows, adjusted_p):
            row["p_value_BH"] = p_value
        coefficient_rows.extend(outcome_rows)

        l1_terms = [index for index, term in enumerate(model.params.index) if term.startswith("C(l1)")]
        restriction = np.zeros((len(l1_terms), len(model.params)))
        for row_index, parameter_index in enumerate(l1_terms):
            restriction[row_index, parameter_index] = 1.0
        joint_p[outcome] = float(model.wald_test(restriction, scalar=True).pvalue)
    diagnostics = pd.DataFrame(
        [
            {"quantity": "Overall OOF MAE", "value": analysis.absolute_error.mean()},
            {"quantity": "Worst-best L1 MAE gap", "value": group_frame.MAE.max() - group_frame.MAE.min()},
            {"quantity": "Signed-error adjusted R-squared", "value": fitted_models["signed_error"].rsquared_adj},
            {"quantity": "Absolute-error adjusted R-squared", "value": fitted_models["absolute_error"].rsquared_adj},
            {"quantity": "Joint L1 p for signed error", "value": joint_p["signed_error"]},
            {"quantity": "Joint L1 p for absolute error", "value": joint_p["absolute_error"]},
            {"quantity": "Number of L1 groups (n>=20)", "value": len(group_frame)},
        ]
    )
    pd.DataFrame(coefficient_rows).to_csv(OUT / "adjusted_l1_coefficients.csv", index=False)
    return group_frame, diagnostics


def create_figures(wi_metrics_frame: pd.DataFrame, test_predictions: pd.DataFrame, groups: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    palette = {"navy": "#244B74", "blue": "#4F81BD", "orange": "#D9822B", "gray": "#6B7280"}

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.scatter(
        test_predictions.wi_score,
        test_predictions.script_score,
        color=palette["blue"],
        alpha=0.72,
        edgecolor="white",
        linewidth=0.4,
    )
    slope, intercept = np.polyfit(test_predictions.wi_score, test_predictions.script_score, 1)
    xs = np.linspace(test_predictions.wi_score.min(), test_predictions.wi_score.max(), 100)
    ax.plot(xs, slope * xs + intercept, color=palette["orange"], linewidth=2)
    ax.set_xlabel("Transferred W&I proficiency score (A=0, B=1, C=2)")
    ax.set_ylabel("Human FCE script score (0–40)")
    ax.set_title("Cross-corpus association on the official FCE test set")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "cross_corpus_transfer.png", dpi=300)
    plt.close(fig)

    plot_groups = groups.sort_values("MAE")
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    xerr = np.vstack(
        [plot_groups.MAE - plot_groups.MAE_CI_low, plot_groups.MAE_CI_high - plot_groups.MAE]
    )
    ax.errorbar(
        plot_groups.MAE,
        plot_groups.L1,
        xerr=xerr,
        fmt="o",
        color=palette["navy"],
        ecolor=palette["gray"],
        capsize=2.5,
    )
    ax.set_xlabel("Out-of-fold mean absolute error (95% bootstrap CI)")
    ax.set_ylabel("First-language code")
    ax.set_title("Error variation across L1 groups")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "l1_subgroup_mae.png", dpi=300)
    plt.close(fig)


def save_profile(wi: pd.DataFrame, fce: pd.DataFrame) -> None:
    profile = pd.DataFrame(
        [
            {"corpus": "W&I", "split": split, "texts_or_scripts": len(part), "writers": part.userid.nunique()}
            for split, part in wi.groupby("eval_split", sort=False)
        ]
        + [
            {"corpus": "FCE", "split": split, "texts_or_scripts": len(part), "writers": len(part)}
            for split, part in fce.groupby("split", sort=False)
        ]
    )
    profile.to_csv(OUT / "data_profile.csv", index=False)


def main() -> None:
    seed_everything()
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    wi = load_wi()
    fce = load_fce()
    save_profile(wi, fce)

    wi_result, wi_bundle = fit_wi_models(wi)
    wi_result.to_csv(OUT / "wi_cefr_metrics.csv", index=False)

    fce_result, test_predictions, fce_scored = fit_fce_holdout(fce, wi_bundle)
    fce_result.to_csv(OUT / "fce_holdout_metrics.csv", index=False)
    test_predictions.drop(columns="text").to_csv(OUT / "fce_test_predictions.csv", index=False)

    chosen_alpha = float(
        fce_result.loc[fce_result.model == "TF-IDF", "alpha"].iloc[0]
    )
    oof = oof_tfidf_predictions(fce_scored, alpha=chosen_alpha)
    oof.drop(columns="text").to_csv(OUT / "fce_oof_predictions.csv", index=False)
    groups, diagnostics = subgroup_audit(oof)
    groups.to_csv(OUT / "l1_subgroup_metrics.csv", index=False)
    diagnostics.to_csv(OUT / "audit_diagnostics.csv", index=False)

    create_figures(wi_result, test_predictions, groups)
    summary = {
        "seed": SEED,
        "wi_records": len(wi),
        "fce_scripts": len(fce),
        "wi_metrics": wi_result.to_dict(orient="records"),
        "fce_metrics": fce_result.to_dict(orient="records"),
        "audit": diagnostics.to_dict(orient="records"),
    }
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    summary = json_safe(summary)
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
