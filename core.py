"""
core.py — training + test pipeline for the CNN-LSTM ablation study.

This replaces the two old scripts (training file + test file):
    train_one()     = old training script  (Phases 1-8)
    evaluate_one()  = old test pipeline    (Steps 1-6)
Both are driven by run_all.py, one (variant, seed, dataset) at a time.
Figures/SHAP live in figures.py and run only for the runs you ask for.
"""
import os
import random
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# ====================================================================
# CONSTANTS  (same values as the old scripts)
# ====================================================================
RUL_MAX         = 125
SEQUENCE_LENGTH = 30
T_REF           = 518.67
GAMMA_EXP       = 0.2857
TRAIN_EPOCHS    = int(os.environ.get('ABL_EPOCHS', 40))   # override ONLY for smoke tests
WARMUP_EPOCHS   = 10
PATIENCE_LIMIT  = 6
BATCH_SIZE      = 64
CRITICAL_RUL    = 45        # = tau of the penalty loss; "critical zone" in the metrics
CLIP_PRED_TO_RUL_MAX = True # paper says predictions are clipped to [0, RUL_max]

DATA_DIR = os.environ.get('CMAPSS_DIR', '.')          # where train_/test_/RUL_ files are
OUT_DIR  = os.environ.get('ABLATION_OUT', 'ablation_runs')

INDEX_LABELS    = ['engine_id', 'cycle']
SETTING_LABELS  = ['setting_1', 'setting_2', 'setting_3']
SENSOR_LABELS   = [f'sensor_{i}' for i in range(1, 22)]
COMPLETE_HEADERS = INDEX_LABELS + SETTING_LABELS + SENSOR_LABELS

DATASET_CONFIGS = {
    'FD001': {'n_regimes': 1, 'description': '1 condition, 1 fault mode  (simplest)'},
    'FD002': {'n_regimes': 6, 'description': '6 conditions, 1 fault mode'},
    'FD003': {'n_regimes': 1, 'description': '1 condition, 2 fault modes'},
    'FD004': {'n_regimes': 6, 'description': '6 conditions, 2 fault modes (hardest)'},
}

# ====================================================================
# VARIANTS   P = physics, A = attention, D = delta, L = penalty loss
# ====================================================================
VARIANTS = {
    'V1' : dict(name='Baseline',      delta=0, physics=0, attention=0, penalty=0),
    'V2' : dict(name='+P',            delta=0, physics=1, attention=0, penalty=0),
    'V3' : dict(name='+A',            delta=0, physics=0, attention=1, penalty=0),
    'V4' : dict(name='+D',            delta=1, physics=0, attention=0, penalty=0),
    'V5' : dict(name='+L',            delta=0, physics=0, attention=0, penalty=1),
    'V6' : dict(name='Full',          delta=1, physics=1, attention=1, penalty=1),
    'V7' : dict(name='Full - P',      delta=1, physics=0, attention=1, penalty=1),
    'V8' : dict(name='Full - A',      delta=1, physics=1, attention=0, penalty=1),
    'V9' : dict(name='Full - D',      delta=0, physics=1, attention=1, penalty=1),
    'V10': dict(name='Full - L',      delta=1, physics=1, attention=1, penalty=0),
    'V11': dict(name='P+A (optional)', delta=0, physics=1, attention=1, penalty=0),
    # Post-hoc variant, added AFTER seeing the V1-V10 results (delta + penalty only).
    'V12': dict(name='Delta + Penalty (post-hoc)', delta=1, physics=0, attention=0, penalty=1),
}
DEFAULT_VARIANTS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']

# Same formulas as the old code. Column names renamed to match the paper.
PHYSICS_REQUIREMENTS = {
    'Phys_Compressor_Efficiency_Proxy': ['sensor_2', 'sensor_4', 'sensor_9', 'sensor_14'],
    'Phys_Corrected_Coolant_Flow'     : ['sensor_2', 'sensor_20'],
    'Phys_Thermal_Load_Ratio'         : ['sensor_4', 'sensor_11'],
    'Phys_Combustion_Load_Index'      : ['sensor_4', 'sensor_9', 'sensor_12'],
    'Phys_Running_Mean_T50'           : ['sensor_4'],
    'Phys_Fan_Bypass_Ratio'           : ['sensor_8', 'sensor_15'],
}


# ====================================================================
# UTILITIES
# ====================================================================
def out_path(*parts):
    p = os.path.join(OUT_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def run_tag(variant, seed, dataset_id):
    return f'{variant}_s{seed}_{dataset_id}'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def compute_metrics(y_true, y_pred):
    """All metrics on the test set. 'crit_*' = engines with true (capped) RUL <= CRITICAL_RUL."""
    d = y_pred - y_true
    out = {
        'n_engines' : int(len(y_true)),
        'rmse'      : float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae'       : float(mean_absolute_error(y_true, y_pred)),
        'nasa'      : nasa_score(y_true, y_pred),
        'unsafe_pct': float(100.0 * np.mean(d > 0)),
        'mean_error': float(np.mean(d)),
    }
    m = y_true <= CRITICAL_RUL
    out['crit_n'] = int(m.sum())
    if m.any():
        out['crit_rmse']       = float(np.sqrt(np.mean(d[m] ** 2)))
        out['crit_unsafe_pct'] = float(100.0 * np.mean(d[m] > 0))
        out['crit_mean_error'] = float(np.mean(d[m]))
    else:
        out['crit_rmse'] = out['crit_unsafe_pct'] = out['crit_mean_error'] = float('nan')
    return out


# ====================================================================
# MODEL
# ====================================================================
class AviationPenaltyLoss(nn.Module):
    def __init__(self, penalty_multiplier=3.0, safety_threshold=45.0):
        super().__init__()
        self.alpha = penalty_multiplier
        self.threshold = safety_threshold

    def forward(self, predictions, targets):
        sq_err = (predictions - targets) ** 2
        mask = (targets <= self.threshold) & (predictions > targets)
        w = torch.where(mask,
                        torch.full_like(predictions, self.alpha),
                        torch.ones_like(predictions))
        return torch.mean(sq_err * w)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        w = torch.softmax(self.fc(lstm_out), dim=1)
        return (w * lstm_out).sum(dim=1), w


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.block(x.permute(0, 2, 1)).permute(0, 2, 1)


class CNNLSTMSequential(nn.Module):
    """use_attention is now a constructor argument (old code read a global flag)."""
    def __init__(self, input_size, use_attention=True,
                 cnn_channels=64, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.cnn = CNNBlock(input_size, cnn_channels)
        self.lstm = nn.LSTM(cnn_channels, hidden_size, num_layers,
                            batch_first=True, dropout=0.3)
        self.attn = TemporalAttention(hidden_size) if use_attention else None
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1),
        )

    def forward(self, x):
        c = self.cnn(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(c, (h0, c0))
        if self.use_attention:
            ctx, _ = self.attn(out)
        else:
            ctx = out[:, -1, :]
        return self.head(ctx)


# ====================================================================
# FEATURE PIPELINE  (shared by training and test => no drift between them)
# ====================================================================
def detect_dead_sensors(df, sensor_labels, threshold=1e-8):
    """NOTE: threshold is applied to the VARIANCE (as in the old code)."""
    var = df[sensor_labels].var()
    return list(var[var < threshold].index)


def assign_flight_regimes(train_df, val_df, setting_labels, n_regimes):
    train_df, val_df = train_df.copy(), val_df.copy()
    if n_regimes == 1:
        train_df['flight_regime'] = 0
        val_df['flight_regime'] = 0
        return train_df, val_df, None
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    train_df['flight_regime'] = kmeans.fit_predict(train_df[setting_labels])
    val_df['flight_regime'] = kmeans.predict(val_df[setting_labels])
    return train_df, val_df, kmeans


def delta_for_frame(df, active_sensors, global_regime_mean):
    """Delta = sensor - engine's own healthy baseline (first 20 cycles per engine+regime).
    Same logic the old training and test code each had inline."""
    df = df.copy().sort_values(['engine_id', 'cycle'])
    baseline = (df[df['cycle'] <= 20]
                .groupby(['engine_id', 'flight_regime'])[active_sensors]
                .mean().reset_index())
    baseline.columns = (['engine_id', 'flight_regime'] +
                        [f'{s}_base' for s in active_sensors])
    df = df.merge(baseline, on=['engine_id', 'flight_regime'], how='left')
    for s in active_sensors:
        df[f'{s}_base'] = df[f'{s}_base'].fillna(
            df['flight_regime'].map(global_regime_mean[s]))
        df[f'delta_{s}'] = df[s] - df[f'{s}_base']
        df.drop(columns=[f'{s}_base'], inplace=True)
    return df


def compute_physics_features(df, active_sensors, features=None):
    """features=None -> every feature whose sensors are active (training).
    features=[...]  -> exactly those (test: guarantees the training feature set)."""
    df = df.copy().sort_values(['engine_id', 'cycle'])
    wanted = list(PHYSICS_REQUIREMENTS) if features is None else list(features)
    computed = []
    for feat in wanted:
        if not all(s in active_sensors for s in PHYSICS_REQUIREMENTS[feat]):
            continue
        if feat == 'Phys_Compressor_Efficiency_Proxy':
            T2, T3 = df['sensor_2'], df['sensor_4']
            ratio = df['sensor_14'] / (df['sensor_9'] + 1e-5)
            T3_ideal = T2 * (ratio.clip(lower=0.01) ** GAMMA_EXP)
            df[feat] = (T3_ideal - T2) / (T3 - T2 + 1e-5)
        elif feat == 'Phys_Corrected_Coolant_Flow':
            theta = df['sensor_2'] / T_REF
            df[feat] = df['sensor_20'] / np.sqrt(theta.clip(lower=0.01))
        elif feat == 'Phys_Thermal_Load_Ratio':
            df[feat] = df['sensor_4'] / (df['sensor_11'] + 1e-5)
        elif feat == 'Phys_Combustion_Load_Index':
            df[feat] = (df['sensor_4'] * df['sensor_9']) / (df['sensor_12'] + 1e-5)
        elif feat == 'Phys_Running_Mean_T50':
            df[feat] = (df.groupby('engine_id')['sensor_4']
                        .transform(lambda x: x.expanding().mean()))
        elif feat == 'Phys_Fan_Bypass_Ratio':
            df[feat] = df['sensor_8'] / (df['sensor_15'] + 1e-5)
        computed.append(feat)
    return df, computed


def create_temporal_sequences(data, feature_cols, sequence_length=30,
                              target_col='Target_RUL', max_sequences=15):
    X_seqs, y_targets = [], []
    for eid in data['engine_id'].unique():
        eng = data[data['engine_id'] == eid].sort_values('cycle')
        if len(eng) < sequence_length:
            continue
        feats = eng[feature_cols].values
        targets = eng[target_col].values
        max_s = len(eng) - sequence_length
        if max_s == 0:
            X_seqs.append(feats[:sequence_length])
            y_targets.append(targets[sequence_length - 1])
            continue
        n_dist = max_sequences - 1
        starts = (list(range(max_s)) if max_s <= n_dist else
                  [int(round(i * max_s / n_dist)) for i in range(n_dist)])
        starts.append(max_s)
        seen, unique = set(), []
        for s in starts:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        for s in unique:
            X_seqs.append(feats[s: s + sequence_length])
            y_targets.append(targets[s + sequence_length - 1])
    return np.array(X_seqs), np.array(y_targets)


def prepare_training_data(dataset_id, cfg):
    """Old training Phases 1-5, with the delta/physics flags actually applied.
    Deterministic: does not depend on the seed (KMeans random_state=42, fixed 80/20 split)."""
    n_regimes = DATASET_CONFIGS[dataset_id]['n_regimes']
    train_file = os.path.join(DATA_DIR, f'train_{dataset_id}.txt')
    if not os.path.exists(train_file):
        return None
    raw = pd.read_csv(train_file, sep=r'\s+', names=COMPLETE_HEADERS)

    engines = raw['engine_id'].unique()
    split_idx = int(len(engines) * 0.8)
    train_ids, val_ids = engines[:split_idx], engines[split_idx:]

    dead_sensors = detect_dead_sensors(raw[raw['engine_id'].isin(train_ids)], SENSOR_LABELS)
    active_sensors = [s for s in SENSOR_LABELS if s not in dead_sensors]
    cleaned = raw.drop(columns=dead_sensors)

    max_cyc = cleaned.groupby('engine_id')['cycle'].max().reset_index()
    max_cyc.columns = ['engine_id', 'max_life']
    data_df = cleaned.merge(max_cyc, on='engine_id')
    data_df['Target_RUL'] = (data_df['max_life'] - data_df['cycle']).clip(upper=RUL_MAX)
    data_df.drop(columns=['max_life'], inplace=True)

    train_m = data_df[data_df['engine_id'].isin(train_ids)].copy()
    val_m = data_df[data_df['engine_id'].isin(val_ids)].copy()
    train_m, val_m, kmeans = assign_flight_regimes(train_m, val_m, SETTING_LABELS, n_regimes)

    global_regime_mean = train_m.groupby('flight_regime')[active_sensors].mean()

    delta_features, regime_baseline = [], None
    if cfg['delta']:
        train_m = delta_for_frame(train_m, active_sensors, global_regime_mean)
        val_m = delta_for_frame(val_m, active_sensors, global_regime_mean)
        delta_features = [f'delta_{s}' for s in active_sensors]
        regime_baseline = (train_m[train_m['cycle'] <= 20]
                           .groupby(['engine_id', 'flight_regime'])[active_sensors]
                           .mean().reset_index())

    physics_features = []
    if cfg['physics']:
        train_m, physics_features = compute_physics_features(train_m, active_sensors)
        val_m, _ = compute_physics_features(val_m, active_sensors, physics_features)

    train_m[active_sensors] = train_m[active_sensors].astype(float)
    val_m[active_sensors] = val_m[active_sensors].astype(float)
    regime_scalers_pool = {}
    for rid in range(n_regimes):
        m_tr = train_m['flight_regime'] == rid
        m_vl = val_m['flight_regime'] == rid
        if m_tr.any():
            sc = MinMaxScaler()
            train_m.loc[m_tr, active_sensors] = sc.fit_transform(train_m.loc[m_tr, active_sensors])
            regime_scalers_pool[rid] = sc
            if m_vl.any():
                val_m.loc[m_vl, active_sensors] = sc.transform(val_m.loc[m_vl, active_sensors])

    physics_scaler = delta_scaler = None
    if physics_features:
        physics_scaler = StandardScaler()
        train_m[physics_features] = physics_scaler.fit_transform(train_m[physics_features])
        val_m[physics_features] = physics_scaler.transform(val_m[physics_features])
    if delta_features:
        delta_scaler = StandardScaler()
        train_m[delta_features] = delta_scaler.fit_transform(train_m[delta_features])
        val_m[delta_features] = delta_scaler.transform(val_m[delta_features])

    X_columns = active_sensors + physics_features + delta_features
    prep = {
        'cfg': dict(cfg), 'kmeans': kmeans, 'regime_scalers_pool': regime_scalers_pool,
        'physics_scaler': physics_scaler, 'delta_scaler': delta_scaler,
        'global_regime_mean': global_regime_mean, 'regime_baseline': regime_baseline,
        'active_sensors': active_sensors, 'physics_features': physics_features,
        'delta_features': delta_features, 'X_columns': X_columns,
        'setting_labels': SETTING_LABELS, 'dead_sensors': dead_sensors,
        'RUL_MAX': RUL_MAX, 'SEQUENCE_LENGTH': SEQUENCE_LENGTH,
        'T_REF': T_REF, 'GAMMA_EXP': GAMMA_EXP,
        'n_regimes': n_regimes, 'dataset_id': dataset_id,
    }
    return {'train_matrix': train_m, 'val_matrix': val_m, 'X_columns': X_columns,
            'active_sensors': active_sensors, 'physics_features': physics_features,
            'delta_features': delta_features, 'dead_sensors': dead_sensors,
            'n_train_engines': len(train_ids), 'n_val_engines': len(val_ids),
            'preprocessors': prep}


# ====================================================================
# TRAINING  (old training script)
# ====================================================================
def train_one(variant_id, seed, dataset_id, make_figures=False, verbose=True):
    cfg = VARIANTS[variant_id]
    tag = run_tag(variant_id, seed, dataset_id)
    device = get_device()

    d = prepare_training_data(dataset_id, cfg)
    if d is None:
        print(f'  [skip] train_{dataset_id}.txt not found in {DATA_DIR}')
        return None
    X_columns = d['X_columns']
    joblib.dump(d['preprocessors'], out_path('preps', tag + '.pkl'))

    X_tr, y_tr = create_temporal_sequences(d['train_matrix'], X_columns, SEQUENCE_LENGTH)
    X_vl, y_vl = create_temporal_sequences(d['val_matrix'], X_columns, SEQUENCE_LENGTH)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_vl_t = torch.tensor(X_vl, dtype=torch.float32)
    y_vl_t = torch.tensor(y_vl, dtype=torch.float32).unsqueeze(1)

    set_seed(seed)                       # seed controls init + shuffling only
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE,
                        shuffle=True, generator=g)

    model = CNNLSTMSequential(len(X_columns), use_attention=bool(cfg['attention'])).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f'  [{tag}] {len(X_columns)} features '
              f'({len(d["active_sensors"])} sensors + {len(d["physics_features"])} physics '
              f'+ {len(d["delta_features"])} delta) | {n_params:,} params | '
              f'dead sensors: {d["dead_sensors"]}')

    mse_fn, pen_fn = nn.MSELoss(), AviationPenaltyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    mk_sched = lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=2)
    sched = mk_sched()

    best_loss, best_state, best_epoch, patience_cnt = float('inf'), None, 0, 0
    history = []
    epochs_run = 0
    for epoch in range(TRAIN_EPOCHS):
        if epoch == WARMUP_EPOCHS:
            # FIX: the old code compared a penalty-loss val value against the best
            # MSE-loss val value from warmup (different objectives), so the penalty
            # phase could never "win" and early stopping could fire on a stale MSE
            # checkpoint. Reset checkpoint/early-stop/LR-scheduler tracking when
            # phase 2 starts. Applied to ALL variants so every variant gets the same
            # training budget and stopping rule.
            best_loss, patience_cnt = float('inf'), 0
            sched = mk_sched()
        use_pen = bool(cfg['penalty']) and epoch >= WARMUP_EPOCHS
        crit = pen_fn if use_pen else mse_fn

        model.train()
        ep_loss = 0.0
        for bX, by in loader:
            bX, by = bX.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(model(bX), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * bX.size(0)
        tr_loss = ep_loss / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            vl_loss = crit(model(X_vl_t.to(device)), y_vl_t.to(device)).item()
        sched.step(vl_loss)
        history.append({'epoch': epoch + 1, 'objective': 'penalty' if use_pen else 'mse',
                        'train_loss': tr_loss, 'val_loss': vl_loss})
        epochs_run = epoch + 1

        if vl_loss < best_loss:
            best_loss, patience_cnt, best_epoch = vl_loss, 0, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch >= WARMUP_EPOCHS:
            patience_cnt += 1
            if patience_cnt >= PATIENCE_LIMIT:
                break

    model.load_state_dict(best_state)
    model.eval()
    torch.save(best_state, out_path('brains', tag + '.pt'))   # written only when training finished
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_path('history', tag + '.csv'), index=False)

    with torch.no_grad():
        vp = model(X_vl_t.to(device)).cpu().numpy().flatten()
    info = {
        'variant': variant_id, 'variant_name': cfg['name'], 'seed': seed,
        'dataset': dataset_id, 'n_features': len(X_columns),
        'n_sensors': len(d['active_sensors']), 'n_physics': len(d['physics_features']),
        'n_delta': len(d['delta_features']), 'dead_sensors': ' '.join(d['dead_sensors']),
        'n_params': n_params, 'epochs_run': epochs_run, 'best_epoch': best_epoch,
        'n_train_engines': d['n_train_engines'], 'n_val_engines': d['n_val_engines'],
        'val_rmse': float(np.sqrt(mean_squared_error(y_vl, vp))),
        'val_mae': float(mean_absolute_error(y_vl, vp)),
        'val_nasa': nasa_score(y_vl, vp),
    }
    if verbose:
        print(f'  [{tag}] trained {epochs_run} ep (best {best_epoch}) | '
              f'val RMSE {info["val_rmse"]:.2f}')

    if make_figures:
        import figures
        figures.plot_loss_curve(hist_df, dataset_id, tag)
        figures.plot_velocity(model, d['val_matrix'], X_columns, dataset_id,
                              tag + '_val', 'Val Engine', device)
        figures.run_shap_engine_level(model, d['val_matrix'], X_columns, X_tr_t,
                                      d['active_sensors'], dataset_id, tag, device)
        figures.run_shap_fleet_level(model, d['val_matrix'], X_columns, X_tr_t,
                                     d['active_sensors'], dataset_id, tag, device)
    return info


# ====================================================================
# TEST  (old test pipeline)
# ====================================================================
def evaluate_one(variant_id, seed, dataset_id, make_figures=False, verbose=True):
    tag = run_tag(variant_id, seed, dataset_id)
    device = get_device()
    need = [os.path.join(DATA_DIR, f'test_{dataset_id}.txt'),
            os.path.join(DATA_DIR, f'RUL_{dataset_id}.txt'),
            out_path('brains', tag + '.pt'), out_path('preps', tag + '.pkl')]
    missing = [f for f in need if not os.path.exists(f)]
    if missing:
        print(f'  [skip test] missing: {missing}')
        return None

    prep = joblib.load(out_path('preps', tag + '.pkl'))   # everything fitted on TRAIN only
    cfg = prep['cfg']
    X_columns = prep['X_columns']
    active_sensors = prep['active_sensors']
    physics_features = prep['physics_features']
    delta_features = prep['delta_features']

    model = CNNLSTMSequential(len(X_columns), use_attention=bool(cfg['attention'])).to(device)
    model.load_state_dict(torch.load(out_path('brains', tag + '.pt'),
                                     map_location=device, weights_only=True))
    model.eval()

    test_raw = pd.read_csv(os.path.join(DATA_DIR, f'test_{dataset_id}.txt'),
                           sep=r'\s+', names=COMPLETE_HEADERS)
    td = test_raw.drop(columns=prep['dead_sensors']).copy()
    td['flight_regime'] = 0 if prep['kmeans'] is None else \
        prep['kmeans'].predict(td[prep['setting_labels']])

    if delta_features:   # baseline = each test engine's OWN first 20 cycles (as before)
        td = delta_for_frame(td, active_sensors, prep['global_regime_mean'])
    if physics_features:
        td, _ = compute_physics_features(td, active_sensors, physics_features)

    td = td.sort_values(['engine_id', 'cycle']).reset_index(drop=True)
    td[active_sensors] = td[active_sensors].astype(float)
    for rid, sc in prep['regime_scalers_pool'].items():
        m = td['flight_regime'] == rid
        if m.any():
            td.loc[m, active_sensors] = sc.transform(td.loc[m, active_sensors])
    if physics_features:
        td[physics_features] = prep['physics_scaler'].transform(td[physics_features])
    if delta_features:
        td[delta_features] = prep['delta_scaler'].transform(td[delta_features])

    preds, skipped = {}, []
    for eid in sorted(td['engine_id'].unique()):
        eng = td[td['engine_id'] == eid]
        if len(eng) < SEQUENCE_LENGTH:
            skipped.append(int(eid))
            continue
        seq = eng[X_columns].values[-SEQUENCE_LENGTH:]
        t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            p = max(0.0, model(t).item())
        if CLIP_PRED_TO_RUL_MAX:
            p = min(float(RUL_MAX), p)
        preds[int(eid)] = p

    true_rul = pd.read_csv(os.path.join(DATA_DIR, f'RUL_{dataset_id}.txt'),
                           sep=r'\s+', header=None).iloc[:, 0].values
    true_map = {i + 1: float(v) for i, v in enumerate(true_rul)}
    rows = []
    for eid, p in preds.items():
        tvc = min(true_map[eid], RUL_MAX)
        rows.append({'engine_id': eid, 'predicted_RUL': p, 'true_RUL': true_map[eid],
                     'true_RUL_capped': tvc, 'error': p - tvc})
    df_res = pd.DataFrame(rows)
    df_res.to_csv(out_path('predictions', tag + '.csv'), index=False)

    metrics = compute_metrics(df_res['true_RUL_capped'].values, df_res['predicted_RUL'].values)
    metrics['n_skipped'] = len(skipped)
    if verbose:
        print(f'  [{tag}] TEST RMSE {metrics["rmse"]:.2f} | NASA {metrics["nasa"]:.1f} | '
              f'unsafe {metrics["unsafe_pct"]:.1f}% | crit-unsafe {metrics["crit_unsafe_pct"]:.1f}% '
              f'(n={metrics["crit_n"]}) | {metrics["n_engines"]} engines, {len(skipped)} skipped')

    if make_figures:
        import figures
        figures.plot_test_eval(df_res, metrics, dataset_id, tag)
        figures.plot_velocity(model, td, X_columns, dataset_id, tag + '_test',
                              'Test Engine', device)
    return metrics
