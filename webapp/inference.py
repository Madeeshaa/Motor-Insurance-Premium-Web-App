import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class MLPStacker(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(64, 1),
        )
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
    def forward(self, x):
        return self.net(x).squeeze(1)

# --- Global Model Loading ---
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_models.joblib"))

try:
    print(f"Loading models from {MODEL_PATH}...")
    bundle = joblib.load(MODEL_PATH)
    
    nb_model = bundle["nb_model"]
    gamma_model = bundle["gamma_model"]
    
    X_mean = bundle["X_mean"]
    X_std = bundle["X_std"]
    calib_lam = bundle["calib_lam"]
    calib_sev = bundle["calib_sev"]
    n_rules = bundle["N_RULES"]
    
    model_nf = ANFIS(X_mean, X_std, input_dim=5, n_rules=n_rules).to(DEVICE)
    model_nf.load_state_dict(bundle["model_nf_state"])
    model_nf.eval()
    
    mlp_in_dim = bundle["mlp_in_dim"]
    mlp_stacker = MLPStacker(mlp_in_dim).to(DEVICE)
    mlp_stacker.load_state_dict(bundle["mlp_stacker_state"])
    mlp_stacker.eval()
    
    stack_mean = bundle["stack_mean"]
    stack_std = bundle["stack_std"]
    iso = bundle["iso"]
    obs_pure_prem = bundle["obs_pure_prem"]

except Exception as e:
    print(f"Failed to load models. Error: {e}")
    raise e

@torch.no_grad()
def nf_raw(X_np):
    Xn = (X_np - X_mean) / X_std
    xt = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)
    ll, ls = model_nf(xt)
    return ll.cpu().numpy(), ls.cpu().numpy()

@torch.no_grad()
def nf_predict(X_np):
    ll, ls = nf_raw(X_np)
    lam = np.log1p(np.exp(np.clip(ll + calib_lam, -50, 50)))
    sev = np.exp(np.clip(ls, -20, 13)) * calib_sev
    return np.clip(lam, 0, None), np.clip(sev, 0, None)

@torch.no_grad()
def mlp_predict_raw(X_norm):
    xt = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)
    return mlp_stacker(xt).cpu().numpy()

def build_stack_features_v2(glm_freq_v, glm_sev_v, glm_pp_v, nf_freq_v, nf_sev_v, nf_pp_v, raw_feat):
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

import itertools

TERMS = {
    "age": ["Young","Adult","Senior"],
    "exp": ["Novice","Average","Expert"],
    "va":  ["New","Mid","Old"],
    "val": ["Budget","Mid","Luxury"],
    "cc":  ["Small","Medium","High"],
}
KEYS = list(TERMS.keys())
ALL_COMBOS = list(itertools.product(*[TERMS[k] for k in KEYS]))

@torch.no_grad()
def get_rule_info(X_np):
    Xn = (X_np - X_mean) / X_std
    xt = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)
    m = model_nf.fuzz(xt)
    w = model_nf._rule_weights(m)
    best_idx = int(torch.argmax(w, dim=1)[0])
    best_weight = float(w[0, best_idx])
    
    # Calculate inherent risk multiplier for this rule natively from trained parameters
    rule_bias_f = float(model_nf.rule_bias_freq[best_idx])
    rule_bias_s = float(model_nf.rule_bias_sev[best_idx])
    rule_score = float(np.exp(rule_bias_f + rule_bias_s))
    
    return "-".join(ALL_COMBOS[best_idx]), best_weight, rule_score

def _compute_eng_features(age, exp, v_age, v_value, cc):
    log_val      = np.log1p(max(v_value, 0))
    age_exp_r    = age / (exp + 1)
    cc_per_va    = cc / (v_age + 1)
    age_x_va     = age * v_age
    exp_x_logval = exp * log_val
    return [age, exp, v_age, v_value, cc, log_val, age_exp_r, cc_per_va, age_x_va, exp_x_logval]

def predict_pure_premium(age, exp, v_age, v_value, cc):
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
