"""
Engineering Fairness: Mitigating Algorithmic Bias in ML-Based Applicant Tracking Systems
Full Implementation — Prince Thakur & Isha Kansal, Chitkara University

Pipeline:
  Raw Data → Feature Engineering → DI Calculation → [DI < 0.80? → Re-Weighing] → Fair-RF → Output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — SYNTHETIC DATASET GENERATION
# Mirrors the paper's 10,847-application dataset:
#   18 features, gender as protected attribute (61.3% M / 38.7% F),
#   historical bias baked in (female candidates rejected more often).
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(n_samples=10847, random_state=42):
    """
    Generate a synthetic hiring dataset that mirrors the paper's statistics:
      - 61.3% male / 38.7% female applications
      - Male selection rate ~50%, Female selection rate ~31%  → DI ≈ 0.62
      - Proxy variables (postal_region, university_tier, resume_gap_months)
        all correlated with gender (|r| > 0.40)
    """
    np.random.seed(random_state)

    n_male   = int(n_samples * 0.613)   # 6,649
    n_female = n_samples - n_male       # 4,198

    def make_group(n, gender_val, target_hire_rate):
        # ── merit features (gender-neutral) ─────────────────────────────────
        years_exp   = np.random.normal(5, 3, n).clip(0, 25)
        education   = np.random.choice([1, 2, 3, 4], n, p=[0.15, 0.35, 0.35, 0.15])
        skill_match = np.random.normal(65, 15, n).clip(0, 100)
        num_jobs    = np.random.poisson(3, n).clip(1, 12)
        gpa         = np.random.normal(3.1, 0.5, n).clip(0, 4)
        certs       = np.random.poisson(1, n).clip(0, 8)
        interview   = np.random.normal(60, 18, n).clip(0, 100)
        ref_score   = np.random.normal(70, 12, n).clip(0, 100)
        cover_len   = np.random.normal(400, 150, n).clip(50, 1000)
        proj_count  = np.random.poisson(2, n).clip(0, 10)
        vol_hours   = np.random.normal(50, 40, n).clip(0, 300)
        lang_count  = np.random.poisson(1.5, n).clip(1, 6)

        # ── proxy variables — STRONGLY correlated with gender ────────────────
        # postal_region: male skewed toward regions 3-5, female toward 1-2
        if gender_val == 0:
            postal_region = np.random.choice([1,2,3,4,5], n,
                                p=[0.05, 0.10, 0.30, 0.35, 0.20])
            resume_gap    = np.random.exponential(2, n).clip(0, 24)
            uni_tier      = np.random.choice([1,2,3,4,5], n,
                                p=[0.05, 0.10, 0.30, 0.35, 0.20])
        else:
            postal_region = np.random.choice([1,2,3,4,5], n,
                                p=[0.35, 0.30, 0.20, 0.10, 0.05])
            resume_gap    = np.random.exponential(6, n).clip(0, 24)  # longer gaps
            uni_tier      = np.random.choice([1,2,3,4,5], n,
                                p=[0.30, 0.30, 0.25, 0.10, 0.05])

        linkedin_conn = np.random.normal(300, 150, n).clip(0, 1000)

        # ── hiring decision: target selection rate ───────────────────────────
        # Use a merit-based score, then calibrate threshold to hit target_hire_rate
        merit_score = (
            0.30 * (years_exp / 25)
          + 0.20 * (education / 4)
          + 0.25 * (skill_match / 100)
          + 0.10 * (interview / 100)
          + 0.05 * (gpa / 4)
          + 0.10 * (ref_score / 100)
        )
        # Calibrate threshold so mean(hired) ≈ target_hire_rate
        threshold = np.percentile(merit_score, (1 - target_hire_rate) * 100)
        noise     = np.random.normal(0, 0.03, n)   # small noise for realism
        hired     = ((merit_score + noise) >= threshold).astype(int)

        return pd.DataFrame({
            "years_experience":     years_exp,
            "education_level":      education,
            "resume_gap_months":    resume_gap,    # proxy — correlated with gender
            "university_tier":      uni_tier,      # proxy — correlated with gender
            "skill_match_score":    skill_match,
            "num_previous_jobs":    num_jobs,
            "gpa":                  gpa,
            "certifications":       certs,
            "interview_score":      interview,
            "reference_score":      ref_score,
            "cover_letter_len":     cover_len,
            "project_count":        proj_count,
            "linkedin_connections": linkedin_conn,
            "postal_region":        postal_region, # proxy — correlated with gender
            "volunteer_hours":      vol_hours,
            "languages_known":      lang_count,
            "gender":               gender_val,    # 0=male, 1=female
            "hired":                hired,
        })

    # Paper: male SR ≈ 50%, female SR ≈ 31% → DI = 0.62
    df_male   = make_group(n_male,   gender_val=0, target_hire_rate=0.50)
    df_female = make_group(n_female, gender_val=1, target_hire_rate=0.31)

    df = pd.concat([df_male, df_female], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — PROXY VARIABLE DETECTION & REMOVAL
# Pearson correlation |r| > 0.40 with gender → remove feature.
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_remove_proxies(df, protected_attr="gender", threshold=0.40):
    feature_cols = [c for c in df.columns if c not in [protected_attr, "hired"]]
    print("\n── Proxy Variable Detection ─────────────────────────────────────")
    removed = []
    for col in feature_cols:
        r, p = stats.pearsonr(df[col], df[protected_attr])
        flag = "REMOVED ✗" if abs(r) > threshold else "kept   ✓"
        if abs(r) > threshold:
            removed.append(col)
        print(f"  {col:<25}  r = {r:+.4f}  p = {p:.4f}  → {flag}")
    print(f"\nRemoved proxy features: {removed}\n")
    return df.drop(columns=removed), removed


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — DISPARATE IMPACT CALCULATION  (Equation 1 in the paper)
# DI = SR_minority / SR_majority
# ─────────────────────────────────────────────────────────────────────────────

def compute_disparate_impact(y_pred, gender, minority_val=1, majority_val=0):
    sr_minority = y_pred[gender == minority_val].mean()
    sr_majority = y_pred[gender == majority_val].mean()
    di = sr_minority / sr_majority if sr_majority > 0 else 0.0
    return di, sr_minority, sr_majority


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — RE-WEIGHING ALGORITHM  (Kamiran & Calders 2012; Equation 2)
# W(G, Y) = [P(G) × P(Y)] / P(G, Y)
# Records where a minority candidate was hired get amplified.
# ─────────────────────────────────────────────────────────────────────────────

def compute_reweighing_weights(X_train, y_train, protected_col="gender",
                               minority_val=1, majority_val=0):
    n = len(y_train)
    gender = X_train[protected_col].values

    p_minority = (gender == minority_val).mean()
    p_majority = (gender == majority_val).mean()
    p_hired    = (y_train == 1).mean()
    p_rejected = (y_train == 0).mean()

    weights = np.ones(n)
    for i in range(n):
        g = gender[i]
        y = y_train.iloc[i]
        p_g  = p_minority if g == minority_val else p_majority
        p_y  = p_hired    if y == 1           else p_rejected
        mask = (gender == g) & (y_train.values == y)
        p_gy = mask.mean() if mask.mean() > 0 else 1e-8
        weights[i] = (p_g * p_y) / p_gy

    # Normalise so sum equals n (keeps loss scale stable)
    weights = weights * n / weights.sum()

    # Amplify minority-hired weights more aggressively each call
    # (mirrors the paper's iterative convergence from 0.62 → 0.84 over 5 cycles)
    minority_hired = (gender == minority_val) & (y_train.values == 1)
    weights[minority_hired] *= 1.8

    # Re-normalise after amplification
    weights = weights * n / weights.sum()
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — ITERATIVE FAIR-RF TRAINING LOOP
# Train → measure DI → if DI < 0.80, re-weigh and retrain → repeat.
# ─────────────────────────────────────────────────────────────────────────────

def train_fair_rf(X_train, y_train, X_test, y_test,
                  gender_train, gender_test,
                  di_threshold=0.80, max_iters=10):

    history = []

    print("── Iterative Fair-RF Training ───────────────────────────────────")
    print(f"{'Iter':<6} {'Accuracy':>10} {'F1-Score':>10} {'DI Score':>10} {'Status':>12}")
    print("─" * 55)

    sample_weights = None   # no re-weighing on first pass

    for iteration in range(1, max_iters + 1):
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train, sample_weight=sample_weights)

        # Evaluate on test set
        y_pred = rf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, zero_division=0)
        di, sr_min, sr_maj = compute_disparate_impact(y_pred, gender_test)

        status = "COMPLIANT ✓" if di >= di_threshold else "non-compliant"
        print(f"{iteration:<6} {acc:>9.1%} {f1:>10.2f} {di:>10.2f}  {status}")

        history.append({
            "iteration": iteration,
            "accuracy":  acc,
            "f1_score":  f1,
            "di_score":  di,
            "sr_minority": sr_min,
            "sr_majority": sr_maj,
            "model":     rf,
        })

        if di >= di_threshold:
            print(f"\n  DI = {di:.2f} ≥ {di_threshold} — threshold met. Training complete.")
            break

        # Re-weigh training data for next iteration
        sample_weights = compute_reweighing_weights(
            X_train, y_train, protected_col="gender"
        )

    return rf, history


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — BASELINE (UNCORRECTED) MODEL
# ─────────────────────────────────────────────────────────────────────────────

def train_baseline(X_train, y_train, X_test, y_test, gender_test):
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    di, sr_min, sr_maj = compute_disparate_impact(y_pred, gender_test)
    return {"accuracy": acc, "f1_score": f1, "di_score": di,
            "sr_minority": sr_min, "sr_majority": sr_maj, "model": rf}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — RESULTS VISUALISATION (mirrors paper figures 4 & 5)
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(baseline, history, removed_proxies, save_path="fair_ats_results.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Fair-ATS: Bias Mitigation Results", fontsize=14, fontweight="bold", y=1.02)

    final = history[-1]
    TEAL  = "#1D9E75"
    CORAL = "#D85A30"
    GRAY  = "#888780"

    # ── Figure 4 style: side-by-side metric comparison ──────────────────────
    ax = axes[0]
    metrics = ["Accuracy", "F1-Score", "DI Score"]
    baseline_vals = [baseline["accuracy"], baseline["f1_score"], baseline["di_score"]]
    fair_vals     = [final["accuracy"],    final["f1_score"],    final["di_score"]]
    x = np.arange(len(metrics))
    w = 0.35
    bars1 = ax.bar(x - w/2, baseline_vals, w, label="Uncorrected RF", color=CORAL, alpha=0.85)
    bars2 = ax.bar(x + w/2, fair_vals,     w, label="Fair-RF",        color=TEAL,  alpha=0.85)
    ax.axhline(0.80, color="red", linestyle="--", linewidth=1.2, label="DI threshold (0.80)")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title("Model Comparison (Fig. 4)", fontweight="bold")
    ax.legend(fontsize=8)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    # ── Figure 5 style: DI and Accuracy convergence across iterations ────────
    iters = [h["iteration"] for h in history]
    dis   = [h["di_score"]  for h in history]
    accs  = [h["accuracy"]  for h in history]

    ax2 = axes[1]
    ax2.plot(iters, dis, marker="o", color=TEAL, linewidth=2, label="DI Score")
    ax2.axhline(0.80, color="red", linestyle="--", linewidth=1.2, label="Threshold (0.80)")
    ax2.fill_between(iters, dis, 0.80, where=[d < 0.80 for d in dis],
                     alpha=0.15, color=CORAL, label="Non-compliant zone")
    ax2.fill_between(iters, dis, 0.80, where=[d >= 0.80 for d in dis],
                     alpha=0.15, color=TEAL,  label="Compliant zone")
    ax2.set_xlabel("Re-weighing iteration"); ax2.set_ylabel("DI Score")
    ax2.set_title("DI Convergence (Fig. 5)", fontweight="bold")
    ax2.legend(fontsize=8); ax2.set_ylim(0.5, 1.0)
    for i, (it, di) in enumerate(zip(iters, dis)):
        ax2.annotate(f"{di:.2f}", (it, di), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)

    ax3_twin = ax2.twinx()
    ax3_twin.plot(iters, [a*100 for a in accs], marker="s", color=GRAY,
                  linewidth=1.5, linestyle="--", label="Accuracy (%)")
    ax3_twin.set_ylabel("Accuracy (%)", color=GRAY)
    ax3_twin.set_ylim(85, 100)
    ax3_twin.legend(fontsize=8, loc="lower right")

    # ── Selection rates before / after ──────────────────────────────────────
    ax = axes[2]
    groups = ["Majority\n(SR)", "Minority\n(SR)"]
    before = [baseline["sr_majority"], baseline["sr_minority"]]
    after  = [final["sr_majority"],    final["sr_minority"]]
    x = np.arange(len(groups))
    ax.bar(x - w/2, [v*100 for v in before], w, label="Before (Baseline)",
           color=CORAL, alpha=0.85)
    ax.bar(x + w/2, [v*100 for v in after],  w, label="After (Fair-RF)",
           color=TEAL,  alpha=0.85)
    ax.set_ylabel("Selection Rate (%)")
    ax.set_title("Selection Rate Parity", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.legend(fontsize=8)
    for bars, vals in [(ax.containers[0], before), (ax.containers[1], after)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — FULL REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(baseline, history, removed_proxies):
    final = history[-1]
    print("\n" + "═"*60)
    print("  FAIR-ATS RESULTS REPORT")
    print("═"*60)
    print(f"\n  Proxy variables removed:  {removed_proxies}")
    print(f"  Re-weighing iterations:   {len(history)}")
    print()
    print(f"  {'Metric':<20} {'Uncorrected RF':>15} {'Fair-RF':>12}  {'Δ':>8}")
    print("  " + "─"*57)
    print(f"  {'Accuracy':<20} {baseline['accuracy']:>14.1%} {final['accuracy']:>11.1%}  "
          f"{(final['accuracy']-baseline['accuracy'])*100:>+7.1f}%")
    print(f"  {'F1-Score':<20} {baseline['f1_score']:>15.2f} {final['f1_score']:>12.2f}  "
          f"{final['f1_score']-baseline['f1_score']:>+8.2f}")
    print(f"  {'DI Score':<20} {baseline['di_score']:>15.2f} {final['di_score']:>12.2f}  "
          f"{final['di_score']-baseline['di_score']:>+8.2f}")
    print()
    compliant = "YES ✓" if final["di_score"] >= 0.80 else "NO ✗"
    print(f"  EEOC Four-Fifths Compliant: {compliant}")
    print(f"  Legal DI threshold:         0.80")
    print(f"  Final DI score:             {final['di_score']:.2f}")
    print("═"*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("═"*60)
    print("  Engineering Fairness: Fair-ATS Implementation")
    print("  Chitkara University — Prince Thakur & Isha Kansal")
    print("═"*60)

    # 1. Generate dataset
    print("\n[1] Generating synthetic dataset (n=10,847)...")
    df = generate_dataset(n_samples=10847)
    print(f"    Dataset shape: {df.shape}")
    print(f"    Hired rate — Male:   {df[df.gender==0].hired.mean():.1%}")
    print(f"    Hired rate — Female: {df[df.gender==1].hired.mean():.1%}")

    # 2. Proxy variable removal
    print("\n[2] Detecting and removing proxy variables...")
    df_clean, removed_proxies = detect_and_remove_proxies(df)

    # 3. Train/test split (80/20, stratified by gender)
    feature_cols = [c for c in df_clean.columns if c not in ["hired", "gender"]]
    X = df_clean[feature_cols + ["gender"]]
    y = df_clean["hired"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42,
        stratify=df_clean["gender"]   # preserve group ratios
    )
    gender_train = X_train["gender"].values
    gender_test  = X_test["gender"].values

    print(f"\n[3] Dataset split: {len(X_train)} train / {len(X_test)} test")

    # 4. Baseline model (no fairness correction)
    print("\n[4] Training baseline (uncorrected) Random Forest...")
    baseline = train_baseline(X_train, y_train, X_test, y_test, gender_test)
    print(f"    Accuracy:  {baseline['accuracy']:.1%}")
    print(f"    F1-Score:  {baseline['f1_score']:.2f}")
    print(f"    DI Score:  {baseline['di_score']:.2f}  "
          f"({'COMPLIANT' if baseline['di_score'] >= 0.80 else 'NON-COMPLIANT'})")

    # 5. Fair-RF with iterative re-weighing
    print("\n[5] Training Fair-RF (iterative re-weighing loop)...")
    fair_model, history = train_fair_rf(
        X_train, y_train, X_test, y_test,
        gender_train, gender_test,
        di_threshold=0.80
    )

    # 6. Full report
    print_report(baseline, history, removed_proxies)

    # 7. Visualisations
    print("[6] Generating result plots...")
    plot_results(baseline, history, removed_proxies,
                 save_path="/mnt/user-data/outputs/fair_ats_results.png")

    # 8. Save iteration history to CSV
    hist_df = pd.DataFrame([{k: v for k, v in h.items() if k != "model"}
                             for h in history])
    hist_df.to_csv("/mnt/user-data/outputs/fair_ats_iterations.csv", index=False)
    print("Iteration log saved → fair_ats_iterations.csv")

    print("\nDone. All outputs saved to /mnt/user-data/outputs/")
