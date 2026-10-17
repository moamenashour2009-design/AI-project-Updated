import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core import SEQUENCE_LENGTH, WARMUP_EPOCHS, out_path


def plot_loss_curve(hist_df, dataset_id, tag):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hist_df['epoch'] - 1, hist_df['train_loss'], label='Train Loss', color='steelblue', lw=2)
    ax.plot(hist_df['epoch'] - 1, hist_df['val_loss'], label='Val Loss', color='darkorange', lw=2)
    ax.axvline(WARMUP_EPOCHS, color='gray', ls=':', alpha=0.7,
               label=f'Warmup end (ep {WARMUP_EPOCHS})')
    ax.set_title(f'CNN-LSTM {dataset_id} — Training Loss', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, ls='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path('figures', f'loss_{tag}.png'), dpi=300)
    plt.close(fig)


def plot_test_eval(df_res, metrics, dataset_id, tag):
    y_true, y_pred = df_res['true_RUL_capped'].values, df_res['predicted_RUL'].values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, color='steelblue', edgecolors='navy', s=40)
    lim = [0, max(y_true.max(), y_pred.max()) + 10]
    ax1.plot(lim, lim, 'r--', lw=2, label='Perfect prediction')
    ax1.set_xlabel('True RUL (cycles)'); ax1.set_ylabel('Predicted RUL (cycles)')
    ax1.set_title(f'CNN-LSTM {dataset_id}: Predicted vs True RUL', fontweight='bold')
    ax1.legend(); ax1.grid(True, ls='--', alpha=0.4)
    ax1.text(0.05, 0.92, f'RMSE = {metrics["rmse"]:.2f}\nMAE  = {metrics["mae"]:.2f}',
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax2 = axes[1]
    ax2.hist(df_res['error'], bins=30, color='darkorange', edgecolor='black', alpha=0.75)
    ax2.axvline(0, color='red', ls='--', lw=2, label='Zero error')
    ax2.axvline(df_res['error'].mean(), color='blue', lw=2,
                label=f'Mean = {df_res["error"].mean():.2f}')
    ax2.set_xlabel('Error (Predicted − True)'); ax2.set_ylabel('Number of Engines')
    ax2.set_title(f'{dataset_id}: Error Distribution\n(Positive = Overestimate = Unsafe)',
                  fontweight='bold')
    ax2.legend(); ax2.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path('figures', f'test_eval_{tag}.png'), dpi=300)
    plt.close(fig)


def plot_velocity(model, frame, X_columns, dataset_id, tag, label, device):
    """Trajectory + degradation velocity for the engine with the most cycles.
    Also saves the velocity/state table so velocity can be reported numerically."""
    try:
        counts = frame.groupby('engine_id')['cycle'].max()
        best = counts.idxmax()
        eng = frame[frame['engine_id'] == best].sort_values('cycle')
        seqs = [eng[X_columns].values[i: i + SEQUENCE_LENGTH]
                for i in range(len(eng) - SEQUENCE_LENGTH + 1)]
        if not seqs:
            raise RuntimeError('not enough cycles')
        model.eval()
        with torch.no_grad():
            traj = model(torch.tensor(np.array(seqs), dtype=torch.float32)
                         .to(device)).cpu().numpy().flatten()
        smoothed = pd.Series(traj).rolling(window=5, min_periods=1).mean().values
        velocity = np.insert(np.diff(smoothed), 0, 0.0)
        state = np.where(velocity < -1.2, 'Accelerating failure',
                         np.where(velocity > -0.2, 'Stable', 'Standard degradation'))
        cycles = np.arange(SEQUENCE_LENGTH, SEQUENCE_LENGTH + len(traj))
        pd.DataFrame({'cycle': cycles, 'raw_rul': traj, 'smoothed_rul': smoothed,
                      'velocity': velocity, 'state': state}).to_csv(
            out_path('figures', f'velocity_{tag}.csv'), index=False)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(cycles, traj, label='Raw Predicted RUL', color='darkorange',
                alpha=0.4, linestyle='--')
        ax.plot(cycles, smoothed, label='Smoothed Health Trend', color='red', linewidth=3)
        ax.set_title(f'CNN-LSTM {dataset_id} — {label} {best} Health Trajectory',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Operational Flight Cycles'); ax.set_ylabel('Remaining Useful Life (cycles)')
        ax.grid(True, linestyle='--', alpha=0.5); ax.legend()
        fig.tight_layout()
        fig.savefig(out_path('figures', f'velocity_{tag}.png'), dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f'  ⚠️  velocity skipped ({tag}): {e}')


def _shap_values(model, bg_t, smp_t):
    import shap
    model.eval()
    # cuDNN RNN backward needs train mode; disable cuDNN for the SHAP gradient pass only.
    with torch.backends.cudnn.flags(enabled=False):
        explainer = shap.GradientExplainer(model, bg_t)
        raw = explainer.shap_values(smp_t)
    arr = np.array(raw[0] if isinstance(raw, list) else raw)
    if arr.ndim == 4:
        arr = arr.squeeze(-1) if arr.shape[-1] == 1 else arr.mean(-1)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    return arr


def run_shap_engine_level(model, val_matrix, X_columns, X_tr_t, active_sensors,
                          dataset_id, tag, device):
    """Engine-level SHAP: last 10 sequences of the FIRST validation engine (not cherry-picked).
    NOTE: values are averaged over the time axis (signed) before taking |.|, as in the old code."""
    try:
        import shap
        sample_eid = int(val_matrix['engine_id'].iloc[0])
        eng = val_matrix[val_matrix['engine_id'] == sample_eid].sort_values('cycle')
        seqs = [eng[X_columns].values[i: i + SEQUENCE_LENGTH]
                for i in range(len(eng) - SEQUENCE_LENGTH + 1)]
        if not seqs:
            raise RuntimeError('Not enough sequences for engine-level SHAP')
        bg_idx = np.random.choice(len(X_tr_t), size=min(100, len(X_tr_t)), replace=False)
        bg_t = X_tr_t[bg_idx].to(device)
        smp_t = torch.tensor(np.array(seqs[-min(10, len(seqs)):]),
                             dtype=torch.float32).to(device)
        arr = _shap_values(model, bg_t, smp_t)
        vals_2d = arr.mean(axis=1)
        mean_shap = np.abs(vals_2d).mean(axis=0)
        ranked_all = sorted(zip(X_columns, mean_shap), key=lambda x: x[1], reverse=True)
        pd.DataFrame(ranked_all, columns=['feature', 'mean_abs_shap']).to_csv(
            out_path('figures', f'shap_engine_all_{tag}.csv'), index=False)

        s_mask = np.array([c in active_sensors for c in X_columns])
        s_vals = vals_2d[:, s_mask]
        s_imp = np.abs(s_vals).mean(axis=0)
        s_names = np.array(X_columns)[s_mask]
        top5 = np.argsort(s_imp)[::-1][:5]
        shap.summary_plot(s_vals[:, top5], feature_names=[s_names[i] for i in top5],
                          plot_type='bar', show=False)
        plt.title(f'Top 5 Sensor SHAP — Engine {sample_eid} ({dataset_id})',
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_path('figures', f'shap_engine_{tag}.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f'  ⚠️  Engine-level SHAP skipped ({tag}): {e}')


def run_shap_fleet_level(model, val_matrix, X_columns, X_tr_t, active_sensors,
                         dataset_id, tag, device):
    """Fleet-level SHAP over ALL validation engines (last 5 sequences each)."""
    try:
        import shap  # noqa: F401
        bg_idx = np.random.choice(len(X_tr_t), size=min(100, len(X_tr_t)), replace=False)
        bg_t = X_tr_t[bg_idx].to(device)
        s_mask = np.array([c in active_sensors for c in X_columns])
        all_importance, processed = [], 0
        for eng_id in val_matrix['engine_id'].unique():
            eng = val_matrix[val_matrix['engine_id'] == eng_id].sort_values('cycle')
            if len(eng) < SEQUENCE_LENGTH:
                continue
            seqs = [eng[X_columns].values[i: i + SEQUENCE_LENGTH]
                    for i in range(len(eng) - SEQUENCE_LENGTH + 1)]
            n = min(5, len(seqs))
            t = torch.tensor(np.array(seqs[-n:]), dtype=torch.float32).to(device)
            try:
                arr = _shap_values(model, bg_t, t)
            except Exception:
                continue
            per_feature = np.abs(arr.mean(axis=1)).mean(axis=0)
            all_importance.append(per_feature[s_mask])
            processed += 1
        if not all_importance:
            print(f'  ⚠️  No engines produced valid SHAP values ({tag})')
            return
        g_imp, g_std = np.mean(all_importance, axis=0), np.std(all_importance, axis=0)
        s_names = np.array(X_columns)[s_mask]
        ranked = sorted(zip(s_names, g_imp, g_std), key=lambda x: x[1], reverse=True)
        pd.DataFrame(ranked, columns=['sensor', 'mean_abs_shap', 'std']).to_csv(
            out_path('figures', f'shap_fleet_{tag}.csv'), index=False)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh([r[0] for r in ranked[::-1]], [r[1] for r in ranked[::-1]],
                xerr=[r[2] for r in ranked[::-1]], color='steelblue', edgecolor='black',
                linewidth=0.5, error_kw=dict(ecolor='black', capsize=4, linewidth=1.2), alpha=0.85)
        ax.set_xlabel('Mean |SHAP Value| across val engines')
        ax.set_title(f'{dataset_id} Fleet-Level Global Sensor Attribution — {processed} Engines',
                     fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.text(0.98, 0.02, f'n = {processed} engines\n±1 std dev', transform=ax.transAxes,
                fontsize=9, ha='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        fig.tight_layout()
        fig.savefig(out_path('figures', f'shap_fleet_{tag}.png'), dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f'  ⚠️  Fleet-level SHAP skipped ({tag}): {e}')
