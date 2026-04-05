# coding: utf-8
"""
=============================================================================
MOTOR INSURANCE PREMIUM CALCULATION
Fuzzy Logic + Neuro-Fuzzy (ANFIS) + MLP Neural Stacker
=============================================================================
Improvements over previous version:
  * Richer feature engineering (log-value, interaction ratios)
  * Interaction terms in GLM formulas
  * Higher RANK_WEIGHT (0.25) for stronger Gini optimisation
  * More training epochs (100) with extended patience
  * OOF (5-fold) generation of GLM+NF features for MLP stacker
  * MLP Neural Stacker (256->128->64->1) trained with Huber + ranking loss
  * All models saved via joblib for fast Streamlit reload
=============================================================================
"""
import warnings
warnings.filterwarnings("ignore")

import os, time, itertools, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import mean_absolute_error, mean_squared_error, auc
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split, KFold

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n" + "=" * 70)
print("MOTOR INSURANCE PREMIUM MODEL  - GLM + ANFIS + MLP Stacker")
print(f"Device: {DEVICE}")
print("=" * 70 + "\n")

MODEL_SAVE_PATH = "saved_models.joblib"

# =============================================================================
# SECTION 1 - DATA PREPARATION
# =============================================================================
print("=" * 70)
print("SECTION 1 - DATA PREPARATION")
print("=" * 70)

data = pd.read_csv("Motor vehicle insurance data.csv", delimiter=";", low_memory=False)
print(f"Loaded {len(data):,} rows x {data.shape[1]} columns")

for col in ["Date_start_contract", "Date_birth", "Date_driving_licence"]:
    data[col] = pd.to_datetime(data[col], dayfirst=True, errors="coerce")

ref_year = data["Date_start_contract"].dt.year
data["Policyholder_age"]   = (ref_year - data["Date_birth"].dt.year).clip(18, 85)
data["Driving_Experience"] = (ref_year - data["Date_driving_licence"].dt.year).clip(0, 60)
data["Vehicle_age"]        = (ref_year - data["Year_matriculation"]).clip(0, 50)

# --- Engineered features ---
data["log_Value_vehicle"]       = np.log1p(data["Value_vehicle"].clip(0))
data["age_exp_ratio"]           = data["Policyholder_age"] / (data["Driving_Experience"] + 1)
data["cc_per_veh_age"]          = data["Cylinder_capacity"] / (data["Vehicle_age"] + 1)
data["age_x_veh_age"]           = data["Policyholder_age"] * data["Vehicle_age"]
data["exp_x_log_val"]           = data["Driving_Experience"] * data["log_Value_vehicle"]

FEATURE_COLS = [
    "Policyholder_age", "Driving_Experience", "Vehicle_age",
    "Value_vehicle", "Cylinder_capacity",
    "log_Value_vehicle", "age_exp_ratio", "cc_per_veh_age",
    "age_x_veh_age", "exp_x_log_val",
]
BASE_COLS = ["Policyholder_age", "Driving_Experience", "Vehicle_age",
             "Value_vehicle", "Cylinder_capacity"]
TARGET_COLS = ["N_claims_year", "Cost_claims_year"]

df = data[FEATURE_COLS + TARGET_COLS].copy().dropna()
df = df[df["N_claims_year"] >= 0].copy()
df.reset_index(drop=True, inplace=True)
print(f"After cleaning: {len(df):,} rows")

df_train, df_test = train_test_split(df, test_size=0.20, random_state=42)
df_train = df_train.copy(); df_test = df_test.copy()
print(f"Train/Test split: {len(df_train):,} train, {len(df_test):,} test")

sev_cap = float(np.percentile(df_train.loc[df_train["Cost_claims_year"] > 0, "Cost_claims_year"], 99.5))
df_train["Cost_claims_year"] = df_train["Cost_claims_year"].clip(0, sev_cap)
df_test["Cost_claims_year"]  = df_test["Cost_claims_year"].clip(0, sev_cap)
print(f"Severity capped at 99.5th pct: {sev_cap:,.0f}")

obs_freq      = df_train["N_claims_year"].mean()
sev_mask      = df_train["Cost_claims_year"] > 0
obs_sev       = df_train.loc[sev_mask, "Cost_claims_year"].mean()
obs_pure_prem = obs_freq * obs_sev
print(f"\nTrain | freq={obs_freq:.4f}  sev={obs_sev:,.2f}  pp={obs_pure_prem:,.2f}")

df = df_train.copy()

# =============================================================================
# SECTION 2 - FUZZIFICATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2 - FUZZIFICATION")
print("=" * 70)

FUZZY_DEFS = {
    "Policyholder_age": {
        "unit": "years", "domain": (16, 85),
        "terms": {"Young": (16,25,35), "Adult": (28,42,58), "Senior": (52,65,85)},
    },
    "Driving_Experience": {
        "unit": "years", "domain": (0, 40),
        "terms": {"Novice": (0,2,5), "Average": (3,8,15), "Expert": (12,22,40)},
    },
    "Vehicle_age": {
        "unit": "years", "domain": (0, 30),
        "terms": {"New": (0,2,5), "Mid": (3,7,12), "Old": (9,16,30)},
    },
    "Value_vehicle": {
        "unit": "log1p(EUR)", "domain": (6.0, 11.0),
        "terms": {"Budget": (6.0,7.5,9.0), "Mid": (8.0,9.5,10.5), "Luxury": (9.8,10.8,11.0)},
    },
    "Cylinder_capacity": {
        "unit": "cc", "domain": (600, 4000),
        "terms": {"Small": (600,1000,1400), "Medium": (1100,1600,2100), "High": (1800,2600,4000)},
    },
}

def gaussmf_np(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / (sigma + 1e-9))**2)

COLORS = ["#2196F3", "#FF5722", "#4CAF50"]
LSTYLE = ["-", "--", "-."]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Fuzzy Membership Functions - Motor Insurance Variables", fontsize=13, fontweight="bold")
axf = axes.flatten()
for ai, (var, info) in enumerate(FUZZY_DEFS.items()):
    ax = axf[ai]
    d_lo, d_hi = info["domain"]
    x = np.linspace(d_lo, d_hi, 500)
    for (term, (a, b, c)), col, ls in zip(info["terms"].items(), COLORS, LSTYLE):
        sigma = max(c - a, 1e-3) / 4.0
        mu = gaussmf_np(x, b, sigma)
        ax.plot(x, mu, color=col, linestyle=ls, linewidth=2.2, label=term)
        ax.fill_between(x, mu, alpha=0.08, color=col)
    ax.set_title(f"{var}\n[{info['unit']}]", fontsize=9.5, fontweight="bold")
    ax.set_xlim(d_lo, d_hi); ax.set_ylim(-0.05, 1.30)
    ax.set_xlabel(info["unit"], fontsize=8.5)
    ax.set_ylabel("Degree of membership", fontsize=8)
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.25)
axf[-1].set_visible(False)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("mf_plots.png", dpi=150, bbox_inches="tight"); plt.close()
print("MF plots saved -> mf_plots.png")

# =============================================================================
# SECTION 3 - GLM MODELS (with interaction terms)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3 - GLM MODELS (Negative Binomial + Gamma, with interactions)")
print("=" * 70)

formula_freq = (
    "N_claims_year ~ Policyholder_age + Driving_Experience "
    "+ Vehicle_age + log_Value_vehicle + Cylinder_capacity "
    "+ age_exp_ratio + age_x_veh_age + exp_x_log_val"
)
nb_model = smf.glm(
    formula=formula_freq, data=df,
    family=sm.families.NegativeBinomial()
).fit(disp=False)
print("[GLM Frequency] Negative Binomial fitted")
print(nb_model.summary().tables[1])

sev_df_glm = df[df["Cost_claims_year"] > 0].copy()
formula_sev = (
    "Cost_claims_year ~ Policyholder_age + Vehicle_age "
    "+ log_Value_vehicle + age_x_veh_age + exp_x_log_val"
)
gamma_model = smf.glm(
    formula=formula_sev, data=sev_df_glm,
    family=sm.families.Gamma(link=sm.families.links.log())
).fit(disp=False)
print("\n[GLM Severity] Gamma (log link) fitted")
print(gamma_model.summary().tables[1])

df["glm_freq"]          = nb_model.predict(df)
df["glm_sev"]           = gamma_model.predict(df)
df["glm_pure_premium"]  = df["glm_freq"] * df["glm_sev"]
print(f"\nGLM mean PP: {df['glm_pure_premium'].mean():,.2f}  (obs: {obs_pure_prem:,.2f})")

# =============================================================================
# SECTION 4 - RULE GENERATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4 - FUZZY RULE GENERATION")
print("=" * 70)

TERMS = {
    "age": ["Young","Adult","Senior"],
    "exp": ["Novice","Average","Expert"],
    "va":  ["New","Mid","Old"],
    "val": ["Budget","Mid","Luxury"],
    "cc":  ["Small","Medium","High"],
}
KEYS     = list(TERMS.keys())
ALL_COMBOS = list(itertools.product(*[TERMS[k] for k in KEYS]))
N_RULES  = len(ALL_COMBOS)
print(f"Total rules: {N_RULES}  (3^5 = 243)")

# =============================================================================
# SECTION 5 - NEURO-FUZZY (ANFIS) TRAINING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5 - NEURO-FUZZY TRAINING  (ANFIS, 100 epochs)")
print("=" * 70)

# Use only base 5 features for ANFIS fuzzifier (interpretable)
ANFIS_COLS = BASE_COLS

df_m = df.dropna(subset=ANFIS_COLS + TARGET_COLS).copy()
df_m.reset_index(drop=True, inplace=True)

X_raw = df_m[ANFIS_COLS].astype(float).values
yN    = df_m["N_claims_year"].astype(float).values

sev_m  = df_m[df_m["Cost_claims_year"] > 0].copy()
Xs_raw = sev_m[ANFIS_COLS].astype(float).values
yS_raw = sev_m["Cost_claims_year"].astype(float).values

yS_p99 = float(np.percentile(yS_raw, 99.5))
yS_win = np.clip(yS_raw, 0, yS_p99)
yS_log = np.log1p(yS_win)

X_mean = X_raw.mean(axis=0)
X_std  = X_raw.std(axis=0) + 1e-9

X  = (X_raw  - X_mean) / X_std
Xs = (Xs_raw - X_mean) / X_std

X_tf_tr, X_tf_v, yN_tr, yN_v = train_test_split(X, yN, test_size=0.15, random_state=42)
X_ts_tr, X_ts_v, yS_tr, yS_v = train_test_split(Xs, yS_log, test_size=0.15, random_state=42)

X_t     = torch.tensor(X_tf_tr, dtype=torch.float32)
yN_t    = torch.tensor(yN_tr, dtype=torch.float32)
X_v_f   = torch.tensor(X_tf_v, dtype=torch.float32).to(DEVICE)
yN_v_t  = torch.tensor(yN_v, dtype=torch.float32).to(DEVICE)

Xs_t    = torch.tensor(X_ts_tr, dtype=torch.float32)
ySlog_t = torch.tensor(yS_tr, dtype=torch.float32)
X_v_s   = torch.tensor(X_ts_v, dtype=torch.float32).to(DEVICE)
yS_v_t  = torch.tensor(yS_v, dtype=torch.float32).to(DEVICE)

BATCH_FREQ = 1024; BATCH_SEV = 512
freq_loader = DataLoader(TensorDataset(X_t, yN_t),   batch_size=BATCH_FREQ, shuffle=True)
sev_loader  = DataLoader(TensorDataset(Xs_t,ySlog_t), batch_size=BATCH_SEV,  shuffle=True)
print(f"Freq: {len(X_t):,} train rows | Sev: {len(Xs_t):,} train rows")


class GaussianMF(nn.Module):
    def __init__(self, a, b, c):
        super().__init__()
        self.mean_raw  = nn.Parameter(torch.tensor(float(b)))
        self.sigma_raw = nn.Parameter(torch.tensor(float(max(c-a,1e-3))/4.0))
        self.sp = nn.Softplus()
    def forward(self, x):
        return torch.exp(-0.5*((x-self.mean_raw)/(self.sp(self.sigma_raw)+1e-4))**2)


class Fuzzifier(nn.Module):
    def __init__(self, xm, xs):
        super().__init__()
        self.register_buffer("xm", torch.tensor(xm, dtype=torch.float32))
        self.register_buffer("xs", torch.tensor(xs, dtype=torch.float32))
        def s(v, c): return float((v - xm[c]) / xs[c])
        self.age_young  = GaussianMF(s(16,0), s(25,0), s(35,0))
        self.age_adult  = GaussianMF(s(28,0), s(42,0), s(58,0))
        self.age_senior = GaussianMF(s(52,0), s(65,0), s(85,0))
        self.exp_novice  = GaussianMF(s(0,1),  s(2,1),  s(5,1))
        self.exp_average = GaussianMF(s(3,1),  s(8,1),  s(15,1))
        self.exp_expert  = GaussianMF(s(12,1), s(22,1), s(40,1))
        self.va_new = GaussianMF(s(0,2),  s(2,2),  s(5,2))
        self.va_mid = GaussianMF(s(3,2),  s(7,2),  s(12,2))
        self.va_old = GaussianMF(s(9,2),  s(16,2), s(30,2))
        lv_mean = float(np.log1p(xm[3]))
        lv_std  = float(np.log1p(xm[3]+xs[3]) - lv_mean + 1e-9)
        self.register_buffer("lv_mean", torch.tensor(lv_mean, dtype=torch.float32))
        self.register_buffer("lv_std",  torch.tensor(lv_std,  dtype=torch.float32))
        def sl(v): return float((np.log1p(v)-lv_mean)/(lv_std+1e-9))
        self.val_budget = GaussianMF(sl(500),   sl(3500),  sl(9000))
        self.val_mid    = GaussianMF(sl(7000),  sl(15000), sl(25000))
        self.val_luxury = GaussianMF(sl(20000), sl(35000), sl(53000))
        self.cc_small  = GaussianMF(s(600,4),  s(1000,4), s(1400,4))
        self.cc_medium = GaussianMF(s(1100,4), s(1600,4), s(2100,4))
        self.cc_high   = GaussianMF(s(1800,4), s(2600,4), s(4000,4))

    def forward(self, X):
        age=X[:,0]; exp=X[:,1]; va=X[:,2]; cc=X[:,4]
        val_orig = torch.clamp(X[:,3]*self.xs[3]+self.xm[3], min=0.0)
        val_log  = (torch.log1p(val_orig)-self.lv_mean)/(self.lv_std+1e-8)
        m = {
            "age": torch.stack([self.age_young(age), self.age_adult(age), self.age_senior(age)], dim=1),
            "exp": torch.stack([self.exp_novice(exp), self.exp_average(exp), self.exp_expert(exp)], dim=1),
            "va":  torch.stack([self.va_new(va), self.va_mid(va), self.va_old(va)], dim=1),
            "val": torch.stack([self.val_budget(val_log), self.val_mid(val_log), self.val_luxury(val_log)], dim=1),
            "cc":  torch.stack([self.cc_small(cc), self.cc_medium(cc), self.cc_high(cc)], dim=1),
        }
        for k in m:
            m[k] = m[k] / m[k].sum(dim=1, keepdim=True).clamp_min(1e-8)
        return m


class ANFIS(nn.Module):
    def __init__(self, xm, xs, input_dim=5, n_rules=243):
        super().__init__()
        self.fuzz    = Fuzzifier(xm, xs)
        self.n_rules = n_rules
        self.rule_bias_freq = nn.Parameter(torch.zeros(n_rules))
        self.rule_bias_sev  = nn.Parameter(torch.zeros(n_rules))
        self.lin_freq = nn.Linear(input_dim, 1)
        self.lin_sev  = nn.Linear(input_dim, 1)
        hid = 128
        self.mlp_freq = nn.Sequential(
            nn.Linear(input_dim+n_rules, hid), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(hid, 32), nn.GELU(), nn.Linear(32, 1))
        self.mlp_sev = nn.Sequential(
            nn.Linear(input_dim+n_rules, hid), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(hid, 32), nn.GELU(), nn.Linear(32, 1))
        self.dropout = nn.Dropout(0.25)

    def _rule_weights(self, m):
        B = m["age"].shape[0]
        w = (m["age"][:,:,None,None,None,None] * m["exp"][:,None,:,None,None,None] *
             m["va"][:,None,None,:,None,None] * m["val"][:,None,None,None,:,None] *
             m["cc"][:,None,None,None,None,:]).reshape(B, self.n_rules)
        return w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(self, X):
        m   = self.fuzz(X)
        w   = self._rule_weights(m)
        wd  = self.dropout(w)
        feat = torch.cat([X, wd], dim=1)
        log_lam = torch.clamp(self.lin_freq(X).squeeze(1) + wd @ self.rule_bias_freq + self.mlp_freq(feat).squeeze(1), -10.0, 5.0)
        log_sev = torch.clamp(self.lin_sev(X).squeeze(1) + wd @ self.rule_bias_sev + self.mlp_sev(feat).squeeze(1), -2.0, 10.5)
        return log_lam, log_sev


def pairwise_ranking_loss(pred, target, n_pairs=2048):
    if len(pred) < 2: return torch.tensor(0.0, device=pred.device)
    idx = torch.randint(0, len(pred), (n_pairs, 2), device=pred.device)
    i, j = idx[:,0], idx[:,1]
    dy = target[i] - target[j]; dp = pred[i] - pred[j]
    return F.softplus(-dy * dp / dy.abs().clamp_min(1e-4)).mean()


EPOCHS = 100; LR = 5e-4; WEIGHT_DECAY = 1e-3
L1_WEIGHT = 1e-4; REG_RULE = 1e-6; RANK_WEIGHT = 0.25; PATIENCE = 20

model_nf = ANFIS(X_mean, X_std, input_dim=5, n_rules=N_RULES).to(DEVICE)
opt      = torch.optim.AdamW(model_nf.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched    = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR*0.05)

print(f"\nTraining ANFIS: {EPOCHS} epochs, rank_weight={RANK_WEIGHT}")
best_val = float("inf"); patience_counter = 0; history_nf = []; t0 = time.time()

for ep in range(1, EPOCHS+1):
    model_nf.train()
    fl_list, sl_list = [], []
    for xb, yb in freq_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        log_lam, _ = model_nf(xb)
        lam = F.softplus(log_lam)
        loss = (F.poisson_nll_loss(lam, yb, log_input=False, full=False, reduction="mean")
                + RANK_WEIGHT * pairwise_ranking_loss(lam, yb)
                + REG_RULE*(model_nf.rule_bias_freq.pow(2).mean()+model_nf.rule_bias_sev.pow(2).mean())
                + L1_WEIGHT*(model_nf.rule_bias_freq.abs().mean()+model_nf.rule_bias_sev.abs().mean()))
        if not torch.isnan(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(model_nf.parameters(), 2.0)
            opt.step(); fl_list.append(loss.item())
            
    for xb, yb_log in sev_loader:
        xb, yb_log = xb.to(DEVICE), yb_log.to(DEVICE)
        opt.zero_grad()
        _, log_sev = model_nf(xb)
        loss = (F.smooth_l1_loss(log_sev, yb_log, beta=1.0)
                + RANK_WEIGHT * pairwise_ranking_loss(log_sev, yb_log)
                + REG_RULE*model_nf.rule_bias_sev.pow(2).mean()
                + L1_WEIGHT*model_nf.rule_bias_sev.abs().mean())
        if not torch.isnan(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(model_nf.parameters(), 2.0)
            opt.step(); sl_list.append(loss.item())
            
    sched.step()
    
    model_nf.eval()
    with torch.no_grad():
        val_log_lam, _ = model_nf(X_v_f)
        v_lam = F.softplus(val_log_lam)
        val_f_loss = F.poisson_nll_loss(v_lam, yN_v_t, log_input=False, full=False, reduction="mean") + RANK_WEIGHT * pairwise_ranking_loss(v_lam, yN_v_t)
        
        _, val_log_sev = model_nf(X_v_s)
        val_s_loss = F.smooth_l1_loss(val_log_sev, yS_v_t, beta=1.0) + RANK_WEIGHT * pairwise_ranking_loss(val_log_sev, yS_v_t)
        
    fl = float(np.mean(fl_list)) if fl_list else float("nan")
    sl = float(np.mean(sl_list)) if sl_list else float("nan")
    v_fl = float(val_f_loss.item())
    v_sl = float(val_s_loss.item())
    val_total = v_fl + v_sl
    
    history_nf.append({"epoch": ep, "train_loss": fl+sl, "val_loss": val_total})
    if ep % 10 == 0 or ep == 1:
        print(f"  Ep {ep:03d}/{EPOCHS} | T-loss={fl+sl:.4f} | V-loss={val_total:.4f} | {time.time()-t0:.0f}s")
        
    if val_total < best_val - 1e-4: best_val = val_total; patience_counter = 0
    else: patience_counter += 1
    if patience_counter >= PATIENCE:
        print(f"  Early stop at ep {ep} (patience={PATIENCE})"); break

print(f"Training done in {time.time()-t0:.1f}s")

# =============================================================================
# SECTION 6 - CALIBRATION + OOF STACKING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6 - CALIBRATION + OOF NEURAL STACKING")
print("=" * 70)

model_nf.eval()

@torch.no_grad()
def nf_raw(X_np):
    Xn = (X_np - X_mean) / X_std
    xt = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)
    ll, ls = model_nf(xt)
    return ll.cpu().numpy(), ls.cpu().numpy()

# Calibrate NF
ll_all, ls_all = nf_raw(df_m[ANFIS_COLS].values.astype(np.float32))
lam_all = np.log1p(np.exp(np.clip(ll_all, -50, 50)))
calib_lam = float(np.log(obs_freq+1e-12) - np.log(lam_all.mean()+1e-12))

ls_sev_cal = nf_raw(sev_m[ANFIS_COLS].values.astype(np.float32))[1]
pred_sev_mean = float(np.exp(np.clip(ls_sev_cal, -20, 13)).mean())
calib_sev = float(obs_sev / (pred_sev_mean + 1e-8))
print(f"NF calib: freq_shift={calib_lam:+.4f}  sev_factor={calib_sev:.4f}")

@torch.no_grad()
def nf_predict(X_np):
    ll, ls = nf_raw(X_np)
    lam = np.log1p(np.exp(np.clip(ll + calib_lam, -50, 50)))
    sev = np.exp(np.clip(ls, -20, 13)) * calib_sev
    return np.clip(lam, 0, None), np.clip(sev, 0, None)

# --- Full-train predictions ---
X_all_raw = df_m[ANFIS_COLS].values.astype(np.float32)
nf_lam_tr, nf_sev_tr = nf_predict(X_all_raw)
df_m["nf_freq"]         = nf_lam_tr
df_m["nf_sev"]          = nf_sev_tr
df_m["nf_pure_premium"] = nf_lam_tr * nf_sev_tr
df_m["glm_freq"]        = nb_model.predict(df_m)
df_m["glm_sev"]         = gamma_model.predict(df_m)
df_m["glm_pure_premium"]= df_m["glm_freq"] * df_m["glm_sev"]
df_m["obs_loss"]        = df_m["Cost_claims_year"].astype(float)

obs_v  = df_m["obs_loss"].values
glm_p  = df_m["glm_pure_premium"].values
nf_p   = df_m["nf_pure_premium"].values

# Gini function
def gini_index(y_true, y_pred):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) < 2 or yt.sum() <= 0: return 0.0
    order = np.argsort(yp)
    yt_s = yt[order]
    cum_pop  = np.arange(1, len(yt_s)+1) / len(yt_s)
    cum_loss = np.cumsum(yt_s) / (yt_s.sum()+1e-12)
    return float(1.0 - 2.0 * auc(cum_pop, cum_loss))

# --- Linear ensemble sweep ---
best_alpha = 0.5; best_ens_g = -1.0
for a in np.linspace(0, 1, 201):
    g = gini_index(obs_v, a * glm_p + (1-a) * nf_p)
    if g > best_ens_g: best_ens_g = g; best_alpha = a
df_m["ens_pure_premium"] = best_alpha * glm_p + (1-best_alpha) * nf_p
print(f"\nLinear ensemble: alpha(GLM)={best_alpha:.2f}  Gini={best_ens_g:.4f}")

# ---- Build stacker feature matrix ----
def build_stack_features_v2(glm_freq_v, glm_sev_v, glm_pp_v, nf_freq_v, nf_sev_v, nf_pp_v, raw_feat):
    """Rich feature set: model outputs + log transforms + raw features."""
    glm_pp_v  = np.clip(glm_pp_v, 0, None)
    nf_pp_v   = np.clip(nf_pp_v,  0, None)
    glm_freq_v= np.clip(glm_freq_v,0, None)
    glm_sev_v = np.clip(glm_sev_v, 0, None)
    nf_freq_v = np.clip(nf_freq_v, 0, None)
    nf_sev_v  = np.clip(nf_sev_v,  0, None)
    return np.column_stack([
        glm_pp_v, nf_pp_v,
        glm_freq_v, glm_sev_v, nf_freq_v, nf_sev_v,
        np.log1p(glm_pp_v), np.log1p(nf_pp_v),
        np.log1p(glm_freq_v), np.log1p(nf_freq_v),
        np.abs(glm_pp_v - nf_pp_v),
        (glm_pp_v / (nf_pp_v + 1e-8)).clip(0, 5),
        raw_feat,
    ])

# --- 5-Fold OOF stacking ---
print("\nGenerating 5-fold OOF predictions for neural stacker ...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_stack_X = np.zeros((len(df_m), 12 + len(FEATURE_COLS)))
raw_feats_all = df_m[FEATURE_COLS].values.astype(np.float32)

for fold, (tr_idx, val_idx) in enumerate(kf.split(df_m)):
    df_tr_f = df_m.iloc[tr_idx]
    df_val_f = df_m.iloc[val_idx]

    # GLM on fold
    nb_oof = smf.glm(formula_freq, data=df_tr_f, family=sm.families.NegativeBinomial()).fit(disp=False)
    sev_df_f = df_tr_f[df_tr_f["Cost_claims_year"] > 0]
    gm_oof   = smf.glm(formula_sev, data=sev_df_f, family=sm.families.Gamma(link=sm.families.links.log())).fit(disp=False)
    gf_v = nb_oof.predict(df_val_f).values
    gs_v = gm_oof.predict(df_val_f).values
    gp_v = gf_v * gs_v

    # NF on fold (use full NF, calibrate on fold)
    X_val_raw = df_val_f[ANFIS_COLS].values.astype(np.float32)
    nf_f_v, nf_s_v = nf_predict(X_val_raw)
    nf_pp_v = nf_f_v * nf_s_v

    oof_stack_X[val_idx] = build_stack_features_v2(
        gf_v, gs_v, gp_v, nf_f_v, nf_s_v, nf_pp_v,
        raw_feats_all[val_idx])
    print(f"  Fold {fold+1}/5 done")

oof_stack_X = np.nan_to_num(oof_stack_X, nan=0.0, posinf=0.0, neginf=0.0)

# Normalize stacker input
stack_mean = oof_stack_X.mean(axis=0)
stack_std  = oof_stack_X.std(axis=0) + 1e-9
oof_stack_Xn = (oof_stack_X - stack_mean) / stack_std

# --- MLP Neural Stacker ---
print("\nTraining MLP Neural Stacker ...")
IN_DIM = oof_stack_Xn.shape[1]

class MLPStacker(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(64, 1),
        )
        # Init last layer to small values
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
    def forward(self, x):
        return self.net(x).squeeze(1)

mlp_stacker = MLPStacker(IN_DIM).to(DEVICE)
opt_mlp = torch.optim.AdamW(mlp_stacker.parameters(), lr=3e-4, weight_decay=1e-3)
sched_mlp = CosineAnnealingLR(opt_mlp, T_max=120, eta_min=3e-6)

Xs_nn_tr, Xs_nn_val, ys_nn_tr, ys_nn_val = train_test_split(oof_stack_Xn, ys_raw_log, test_size=0.15, random_state=42)
Xs_nn = torch.tensor(Xs_nn_tr, dtype=torch.float32)
ys_nn = torch.tensor(ys_nn_tr, dtype=torch.float32)
Xs_nn_v = torch.tensor(Xs_nn_val, dtype=torch.float32).to(DEVICE)
ys_nn_v = torch.tensor(ys_nn_val, dtype=torch.float32).to(DEVICE)

ds_nn  = TensorDataset(Xs_nn, ys_nn)
dl_nn  = DataLoader(ds_nn, batch_size=512, shuffle=True)

MLP_EPOCHS = 150; RANK_W_MLP = 0.50; best_mlp_val = float("inf"); pat_mlp = 0
history_mlp = []
t1 = time.time()
for ep in range(1, MLP_EPOCHS+1):
    mlp_stacker.train()
    ep_losses = []
    for xb, yb in dl_nn:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_mlp.zero_grad()
        pred = mlp_stacker(xb)
        loss = (F.smooth_l1_loss(pred, yb, beta=0.5)
                + RANK_W_MLP * pairwise_ranking_loss(pred, yb, n_pairs=1024))
        if not torch.isnan(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(mlp_stacker.parameters(), 2.0)
            opt_mlp.step(); ep_losses.append(loss.item())
    sched_mlp.step()
    
    mlp_stacker.eval()
    with torch.no_grad():
        v_pred = mlp_stacker(Xs_nn_v)
        v_loss = F.smooth_l1_loss(v_pred, ys_nn_v, beta=0.5) + RANK_W_MLP * pairwise_ranking_loss(v_pred, ys_nn_v, n_pairs=1024)
        
    el = float(np.mean(ep_losses)) if ep_losses else float("nan")
    vl = float(v_loss.item())
    history_mlp.append({"epoch": ep, "train_loss": el, "val_loss": vl})
    
    if ep % 20 == 0 or ep == 1:
        print(f"  MLP ep {ep:03d}/{MLP_EPOCHS} | T-loss={el:.3f} | V-loss={vl:.3f} | {time.time()-t1:.0f}s")
    if vl < best_mlp_val - 0.001: best_mlp_val = vl; pat_mlp = 0
    else: pat_mlp += 1
    if pat_mlp >= 30:
        print(f"  MLP early stop at ep {ep}"); break

print(f"MLP stacker done in {time.time()-t1:.1f}s")

# --- MLP stacker predictions on train ---
mlp_stacker.eval()
@torch.no_grad()
def mlp_predict_raw(X_norm):
    """Return raw MLP output (log1p score space) — used for ranking/calibration."""
    xt = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)
    chunks = []; bs = 2048
    for i in range(0, len(xt), bs):
        chunks.append(mlp_stacker(xt[i:i+bs]).cpu().numpy())
    return np.concatenate(chunks)

stack_raw = mlp_predict_raw(oof_stack_Xn)

# Decode log1p -> EUR space, then isotonic calibration to obs_v
iso = IsotonicRegression(out_of_bounds="clip")
stack_eur = np.expm1(np.clip(stack_raw, -2, 13))  # decode from log1p scale
stack_eur = np.clip(stack_eur, 0, None)
iso.fit(stack_eur, obs_v)
stack_cal = iso.predict(stack_eur)
df_m["stack_pure_premium"] = np.clip(stack_cal, 0, None)
print(f"MLP Stack mean PP: {df_m['stack_pure_premium'].mean():,.2f}")

print("\nTrain-set Gini:")
for name, col in [("GLM","glm_pure_premium"),("NF","nf_pure_premium"),
                  ("Ensemble","ens_pure_premium"),("MLP-Stack","stack_pure_premium")]:
    g = gini_index(obs_v, df_m[col].values)
    print(f"  {name:<15}: {g:.4f}{'  [>0.35]' if g>0.35 else ''}")

# =============================================================================
# SECTION 7 - EVALUATION ON TEST SET
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7 - MODEL EVALUATION (TEST SET)")
print("=" * 70)

df_test_m = df_test.dropna(subset=ANFIS_COLS + TARGET_COLS + FEATURE_COLS).copy()
df_test_m.reset_index(drop=True, inplace=True)

# GLM
df_test_m["glm_freq"]         = nb_model.predict(df_test_m)
df_test_m["glm_sev"]          = gamma_model.predict(df_test_m)
df_test_m["glm_pure_premium"] = df_test_m["glm_freq"] * df_test_m["glm_sev"]

# NF
X_test_raw = df_test_m[ANFIS_COLS].values.astype(np.float32)
nf_lam_ts, nf_sev_ts = nf_predict(X_test_raw)
df_test_m["nf_freq"]         = nf_lam_ts
df_test_m["nf_sev"]          = nf_sev_ts
df_test_m["nf_pure_premium"] = nf_lam_ts * nf_sev_ts

# Ensemble
df_test_m["ens_pure_premium"] = best_alpha*df_test_m["glm_pure_premium"] + (1-best_alpha)*df_test_m["nf_pure_premium"]

# MLP Stacker
raw_feat_test = df_test_m[FEATURE_COLS].values.astype(np.float32)
st_X_test = build_stack_features_v2(
    df_test_m["glm_freq"].values, df_test_m["glm_sev"].values,
    df_test_m["glm_pure_premium"].values,
    nf_lam_ts, nf_sev_ts, df_test_m["nf_pure_premium"].values,
    raw_feat_test)
st_X_test = np.nan_to_num(st_X_test, nan=0.0, posinf=0.0, neginf=0.0)
st_X_test_n = (st_X_test - stack_mean) / stack_std
stack_raw_test = mlp_predict_raw(st_X_test_n)
stack_eur_test = np.clip(np.expm1(np.clip(stack_raw_test, -2, 13)), 0, None)
st_cal_test = iso.predict(stack_eur_test)
df_test_m["stack_pure_premium"] = np.clip(st_cal_test, 0, None)

df_test_m["obs_loss"] = df_test_m["Cost_claims_year"].astype(float)
obs_v_test = df_test_m["obs_loss"].values
cap_obs = float(np.percentile(obs_v[obs_v > 0], 99.5))
y_eval  = np.clip(obs_v_test, 0, cap_obs)
df_eval = df_test_m


def lorenz_curve(y_true, y_pred):
    yt, yp = np.asarray(y_true,float), np.asarray(y_pred,float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) < 2 or yt.sum() <= 0: return np.array([0.,1.]),np.array([0.,1.])
    order = np.argsort(yp); yt_s = yt[order]; n = len(yt_s)
    return np.arange(1,n+1)/n, np.cumsum(yt_s)/(yt_s.sum()+1e-12)


def compute_metrics(y_true, y_pred, name):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2: return {"Model":name,"n":0,"MAE":np.nan,"RMSE":np.nan,"Gini":np.nan,"Mean_Pred":np.nan,"Mean_Obs":np.nan,"Pred_Obs_Ratio":np.nan}
    return {"Model":name,"n":int(mask.sum()),
            "MAE":float(mean_absolute_error(yt,yp)),
            "RMSE":float(np.sqrt(mean_squared_error(yt,yp))),
            "Gini":gini_index(yt,yp),
            "Mean_Pred":float(yp.mean()),"Mean_Obs":float(yt.mean()),
            "Pred_Obs_Ratio":float(yp.mean()/(yt.mean()+1e-12))}


MODELS_EVAL = {
    "GLM": "glm_pure_premium",
    "NF (ANFIS)": "nf_pure_premium",
    "Ensemble": "ens_pure_premium",
    "MLP-Stack": "stack_pure_premium",
}
results_list = []; lorenz_data = {}
for name, col in MODELS_EVAL.items():
    r = compute_metrics(y_eval, df_eval[col].values, name)
    results_list.append(r)
    lorenz_data[name] = lorenz_curve(y_eval, df_eval[col].values)

results_df = pd.DataFrame(results_list).sort_values("Gini", ascending=False)
results_df["Obs_Pure_Premium"] = obs_pure_prem
results_df.to_csv("model_comparison_results.csv", index=False)

print("\n" + "="*72)
print("MODEL COMPARISON TABLE  (sorted by Gini)")
print("="*72)
print(results_df[["Model","Gini","MAE","RMSE","Mean_Pred","Pred_Obs_Ratio","Obs_Pure_Premium"]].to_string(
    index=False, float_format="{:.4f}".format))
print("="*72)

# Lorenz curves
COLORS_LC = {"GLM":"#1565C0","NF (ANFIS)":"#E53935","Ensemble":"#2E7D32","MLP-Stack":"#8E24AA"}
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0,1],[0,1],"k--",linewidth=1.2,label="Random (Gini=0)")
for name,(cp,cl) in lorenz_data.items():
    g   = float(results_df.loc[results_df["Model"]==name,"Gini"].values[0])
    col = COLORS_LC[name]
    ax.plot(cp,cl,color=col,linewidth=2.2,label=f"{name}  (Gini={g:.4f})")
    ax.fill_between(cp,cl,[0]*len(cl),alpha=0.05,color=col)
ax.set_title("Lorenz Curve Comparison - Motor Insurance Models",fontsize=12,fontweight="bold")
ax.set_xlabel("Cumulative proportion of policies",fontsize=10)
ax.set_ylabel("Cumulative proportion of observed loss",fontsize=10)
ax.legend(loc="upper left",fontsize=10); ax.grid(alpha=0.25)
ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig("lorenz_curves.png",dpi=150,bbox_inches="tight"); plt.close()
print("Lorenz curves saved -> lorenz_curves.png")

# Decile tables
def decile_table(score, obs, n=10):
    tmp = pd.DataFrame({"score":score,"obs":obs})
    tmp["decile"] = pd.qcut(tmp["score"].rank(method="first"),n,labels=False)+1
    tab = tmp.groupby("decile").agg(avg_pred=("score","mean"),avg_obs=("obs","mean"),count=("obs","count")).reset_index()
    tab["lift"] = tab["avg_obs"] / (obs.mean()+1e-12)
    tab["residual"] = tab["avg_pred"] - tab["avg_obs"]
    return tab

fig2, axes2 = plt.subplots(2,2,figsize=(14,10))
fig2.suptitle(f"Decile Lift Tables: Predicted vs Observed Pure Premium\n(obs pp = {obs_pure_prem:,.0f})",fontsize=12,fontweight="bold")
for ai,(name,col) in enumerate(MODELS_EVAL.items()):
    dec = decile_table(df_eval[col].values, y_eval)
    ax  = axes2.flatten()[ai]
    x   = dec["decile"].values; clr = COLORS_LC[name]
    ax.bar(x-0.2,dec["avg_pred"],width=0.4,color=clr,alpha=0.85,label="Pred PP")
    ax.bar(x+0.2,dec["avg_obs"], width=0.4,color="grey",alpha=0.55,label="Obs Loss")
    ax.axhline(obs_pure_prem,color="red",linestyle="--",linewidth=1.2,label=f"Obs PP ({obs_pure_prem:,.0f})")
    g = float(results_df.loc[results_df["Model"]==name,"Gini"].values[0])
    ax.set_title(f"{name}  (Gini={g:.4f})",fontsize=10,fontweight="bold")
    ax.set_xlabel("Decile"); ax.set_ylabel("Average amount")
    ax.legend(fontsize=8); ax.grid(alpha=0.2,axis="y")
plt.tight_layout()
plt.savefig("decile_lift.png",dpi=150,bbox_inches="tight"); plt.close()
print("Decile lift chart saved -> decile_lift.png")

print("\n" + "="*72)
print("DECILE TABLES")
print("="*72)
for name, col in MODELS_EVAL.items():
    dec = decile_table(df_eval[col].values, y_eval)
    print(f"\n--- {name} ---")
    print(dec[["decile","count","avg_pred","avg_obs","lift","residual"]].to_string(
        index=False, float_format="{:.2f}".format))

# Survival curves
SCENARIOS = [
    (24,2,2,6000,1000,"Young-Novice-New-Budget"),
    (42,8,7,15000,1600,"Adult-Average-Mid-Mid"),
    (67,22,15,6000,1000,"Senior-Expert-Old-Budget"),
    (42,22,2,45000,2500,"Adult-Expert-New-Luxury"),
]
t_grid = np.linspace(0,5,300)
fig3,ax3 = plt.subplots(figsize=(9,5))
SC_COLORS = ["#2196F3","#F44336","#4CAF50","#FF9800"]
for (age,exp,va,val,cc,label),col in zip(SCENARIOS,SC_COLORS):
    x_raw = np.array([[age,exp,va,val,cc]],dtype=np.float32)
    lam_v,_ = nf_predict(x_raw)
    ax3.plot(t_grid,np.exp(-float(lam_v[0])*t_grid),color=col,linewidth=2,
             label=f"{label}  λ={float(lam_v[0]):.4f}")
ax3.set_title("No-Claim Survival Curves by Fuzzy Scenario",fontsize=11,fontweight="bold")
ax3.set_xlabel("Time (years)"); ax3.set_ylabel("P(no claim up to t)")
ax3.legend(fontsize=8.5,loc="upper right"); ax3.grid(alpha=0.25); ax3.set_ylim(0,1)
plt.tight_layout()
plt.savefig("survival_curves.png",dpi=150,bbox_inches="tight"); plt.close()
print("Survival curves saved -> survival_curves.png")

# =============================================================================
# SECTION 8 - SAVE MODELS
# =============================================================================
print("\n" + "="*70)
print("SECTION 8 - SAVING MODELS")
print("="*70)

save_bundle = {
    "nb_model": nb_model,
    "gamma_model": gamma_model,
    "model_nf_state": model_nf.state_dict(),
    "X_mean": X_mean,
    "X_std": X_std,
    "calib_lam": calib_lam,
    "calib_sev": calib_sev,
    "N_RULES": N_RULES,
    "mlp_stacker_state": mlp_stacker.state_dict(),
    "mlp_in_dim": IN_DIM,
    "stack_mean": stack_mean,
    "stack_std": stack_std,
    "iso": iso,
    "best_alpha": best_alpha,
    "obs_pure_prem": obs_pure_prem,
    "FEATURE_COLS": FEATURE_COLS,
    "ANFIS_COLS": ANFIS_COLS,
    "formula_freq": formula_freq,
    "formula_sev": formula_sev,
}
joblib.dump(save_bundle, MODEL_SAVE_PATH)
# =============================================================================
# SECTION 9 - OVERFITTING ANALYSIS (LEARNING CURVES)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 9 - PLOTTING LEARNING CURVES")
print("=" * 70)
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
# ANFIS
anfis_eps = [h["epoch"] for h in history_nf]
ax1.plot(anfis_eps, [h["train_loss"] for h in history_nf], label="Train Loss", color="blue")
ax1.plot(anfis_eps, [h["val_loss"] for h in history_nf], label="Validation Loss", color="red", linestyle="--")
ax1.set_title("ANFIS Learning Curve")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Combined Loss")
ax1.grid(True, alpha=0.3)
ax1.legend()

# MLP
mlp_eps = [h["epoch"] for h in history_mlp]
ax2.plot(mlp_eps, [h["train_loss"] for h in history_mlp], label="Train Loss", color="blue")
ax2.plot(mlp_eps, [h["val_loss"] for h in history_mlp], label="Validation Loss", color="red", linestyle="--")
ax2.set_title("MLP Stacker Learning Curve")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
print("Learning curves saved -> learning_curves.png")

print("=" * 70)
print(f"Results saved to -> {COMPARE_CSV}")
if MODEL_PATH:
    print(f"Models saved to -> {MODEL_PATH}")
print("PIPELINE COMPLETE.")

# =============================================================================
# SECTION 10 - FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SECTION 10 - FINAL REPORT SUMMARY")
print("="*70)

best_row = results_df.iloc[0]
best_gini_all = float(best_row["Gini"])

print(f"\nGINI PERFORMANCE (Test Set):")
for _, row in results_df.iterrows():
    flag = "[OK >0.35]" if row["Gini"] > 0.35 else "          "
    print(f"  {flag} {row['Model']:<15}: Gini={row['Gini']:.4f} | MAE={row['MAE']:.2f} | RMSE={row['RMSE']:.2f}")

if best_gini_all > 0.35:
    print(f"\n[SUCCESS] Best model '{best_row['Model']}' Gini ({best_gini_all:.4f}) exceeds 0.35 target.")
else:
    print(f"\n[NOTE] Best Gini = {best_gini_all:.4f} (target >=0.35)")

for fname in ["mf_plots.png","lorenz_curves.png","decile_lift.png","survival_curves.png","model_comparison_results.csv",MODEL_SAVE_PATH]:
    mark = "[OK]" if os.path.exists(fname) else "[MISSING]"
    print(f"  {mark}  {fname}")

print("\nPipeline complete.\n")


# =============================================================================
# PREDICTION FUNCTION (for Streamlit UI)
# =============================================================================
def _compute_eng_features(age, exp, v_age, v_value, cc):
    """Compute engineered features from base inputs."""
    log_val      = np.log1p(max(v_value, 0))
    age_exp_r    = age / (exp + 1)
    cc_per_va    = cc / (v_age + 1)
    age_x_va     = age * v_age
    exp_x_logval = exp * log_val
    return [age, exp, v_age, v_value, cc,
            log_val, age_exp_r, cc_per_va, age_x_va, exp_x_logval]


def predict_pure_premium(age, exp, v_age, v_value, cc):
    """Predict pure premium for a single policyholder using the full model stack."""
    eng = _compute_eng_features(age, exp, v_age, v_value, cc)

    # 1. GLM
    df_single = pd.DataFrame([{
        "Policyholder_age": age, "Driving_Experience": exp, "Vehicle_age": v_age,
        "Value_vehicle": v_value, "Cylinder_capacity": cc,
        "log_Value_vehicle": eng[5], "age_exp_ratio": eng[6],
        "cc_per_veh_age": eng[7], "age_x_veh_age": eng[8], "exp_x_log_val": eng[9],
    }])
    glm_freq_val = float(nb_model.predict(df_single).iloc[0])
    glm_sev_val  = float(gamma_model.predict(df_single).iloc[0])
    glm_pp       = glm_freq_val * glm_sev_val

    # 2. NF
    x_base = np.array([[age, exp, v_age, v_value, cc]], dtype=np.float32)
    nf_f, nf_s = nf_predict(x_base)
    nf_pp = float(nf_f[0] * nf_s[0])

    # 3. MLP stacker
    x_full = np.array([eng], dtype=np.float32)
    st_X = build_stack_features_v2(
        np.array([glm_freq_val]), np.array([glm_sev_val]), np.array([glm_pp]),
        nf_f, nf_s, np.array([nf_pp]), x_full)
    st_X = np.nan_to_num(st_X, nan=0.0, posinf=0.0, neginf=0.0)
    st_Xn = (st_X - stack_mean) / stack_std
    raw_pred = mlp_predict_raw(st_Xn)
    final_pp = iso.predict(np.clip(raw_pred, 0, None))[0]
    return float(np.clip(final_pp, 0, None)), glm_pp, nf_pp, float(glm_freq_val), float(nf_f[0])
