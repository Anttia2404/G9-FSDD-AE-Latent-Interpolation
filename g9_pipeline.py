#!/usr/bin/env python3
"""
G9 - Tang cuong du lieu nhan dang chu so noi bang noi suy khong gian tiem an cua AE (FSDD).
Cai dat dung theo de cuong: tien xu ly co dinh, LOSO 6 fold, AE tat dinh (276.001 tham so),
noi suy Linear / SLERP, doi chuan B0/B1/B2, classifier CNN (64.266 tham so).

Lenh:
  python g9_pipeline.py prep      --data free-spoken-digit-dataset/recordings --out out
  python g9_pipeline.py run       --out out                    # TN1 + TN2 day du (6 fold x 3 seed)
  python g9_pipeline.py run       --out out --quick            # chay thu nhanh de kiem tra
  python g9_pipeline.py aggregate --out out
"""
import argparse
import copy
import hashlib
import os
import time
import warnings
from itertools import combinations

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

# ----------------------------------------------------------------------------- hang so (muc 2.2)
SR, N_SAMPLES = 8000, 8000
N_FFT, HOP, N_MEL = 256, 128, 64
T_VALID = 1 + N_SAMPLES // HOP          # 63 khung hop le
PRE_EMPH, EPS = 0.97, 1e-6
ALPHAS = (0.25, 0.50, 0.75)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- tien xu ly
def mel_filterbank():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MEL, fmin=50.0, fmax=4000.0,
                                 htk=False, norm="slaney")
    return fb.astype(np.float64)


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, f"{path}: sr={sr} (can 8000)"
    assert x.shape[1] == 1, f"{path}: khong phai mono"
    return x[:, 0]


def fix_length(v):
    """Dem 0 cuoi neu ngan; neu dai lay doan 8000 mau co nang luong lon nhat (hoa -> doan truoc)."""
    n = len(v)
    if n < N_SAMPLES:
        return np.pad(v, (0, N_SAMPLES - n)), 0, False
    if n == N_SAMPLES:
        return v, 0, False
    c = np.concatenate([[0.0], np.cumsum(v.astype(np.float64) ** 2)])
    sums = c[N_SAMPLES:] - c[:-N_SAMPLES]
    s = int(np.argmax(sums))            # argmax tra ve vi tri dau tien khi bang nhau
    return v[s:s + N_SAMPLES], s, True


def logmel(u, fb):
    u = u.astype(np.float64)
    u = u - u.mean()                                   # DC tren ban ghi chua dem
    v = np.empty_like(u)                               # pre-emphasis a = 0,97
    v[0] = u[0]
    v[1:] = u[1:] - PRE_EMPH * u[:-1]
    v, start, cropped = fix_length(v)
    S = librosa.stft(v, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window="hann",
                     center=True, pad_mode="constant")
    P = fb @ (np.abs(S) ** 2)                          # pho cong suat mel
    out = np.log1p(P).astype(np.float32)               # S = ln(1 + P)
    assert out.shape == (N_MEL, T_VALID), out.shape
    return out, start, cropped


def cmd_prep(a):
    os.makedirs(a.out, exist_ok=True)
    fb = mel_filterbank()
    empty = int((fb.sum(1) == 0).sum())
    files = sorted(f for f in os.listdir(a.data) if f.endswith(".wav"))
    rows, feats = [], []
    for f in files:
        path = os.path.join(a.data, f)
        info = sf.info(path)
        digit, speaker, idx = f[:-4].split("_")
        x = load_wav(path)
        S, start, cropped = logmel(x, fb)
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        rows.append(dict(file=f, label=int(digit), speaker=speaker, index=int(idx),
                         sr=info.samplerate, channels=info.channels, n_samples=info.frames,
                         duration=info.frames / info.samplerate, sha256=sha,
                         cropped=cropped, crop_start=start, silent=bool(np.all(S == 0))))
        feats.append(S)
    meta = pd.DataFrame(rows)
    X = np.stack(feats)
    meta.to_csv(f"{a.out}/manifest.csv", index=False)
    np.save(f"{a.out}/features_raw.npy", X)
    print(f"{len(meta)} ban ghi | {meta.speaker.nunique()} nguoi noi: {sorted(meta.speaker.unique())}")
    print(f"thoi luong: min {meta.duration.min():.3f}s, max {meta.duration.max():.3f}s | "
          f"bi cat: {meta.cropped.mean() * 100:.1f}% | bo loc mel rong (rong): {empty}/{N_MEL}")
    print("phan bo lop:", meta.label.value_counts().sort_index().tolist())
    print("ban ghi rong:", int(meta.silent.sum()), "| trung sha256:", int(meta.sha256.duplicated().sum()))


# ----------------------------------------------------------------------------- chia du lieu (muc 2.3)
def make_splits(meta):
    spk = sorted(meta.speaker.unique())
    assert len(spk) == 6, spk
    splits = []
    for i, te in enumerate(spk):
        va = spk[(i + 1) % 6]
        tr = [s for s in spk if s not in (te, va)]
        splits.append(dict(fold=i + 1, test=te, val=va, train=tr))
    return splits


def norm_tensor(X, mu, sd):
    Z = (X - mu[None, :, None]) / (sd[None, :, None] + EPS)
    Z = np.pad(Z, ((0, 0), (0, 0), (0, 1)))            # cot thu 64 = 0 sau chuan hoa
    return torch.from_numpy(Z[:, None]).float()


# ----------------------------------------------------------------------------- mo hinh (muc 3.2, 3.5)
class AE(nn.Module):
    def __init__(self, dz=64):
        super().__init__()
        c = lambda i, o: nn.Conv2d(i, o, 3, 2, 1)
        t = lambda i, o: nn.ConvTranspose2d(i, o, 3, 2, 1, output_padding=1)
        self.enc = nn.Sequential(c(1, 8), nn.ReLU(), c(8, 16), nn.ReLU(), c(16, 32), nn.ReLU())
        self.fc_e = nn.Linear(2048, dz)
        self.fc_d = nn.Linear(dz, 2048)
        self.dec = nn.Sequential(t(32, 16), nn.ReLU(), t(16, 8), nn.ReLU(), t(8, 1))

    def encode(self, x):
        return self.fc_e(self.enc(x).flatten(1))

    def decode(self, z):
        y = self.dec(F.relu(self.fc_d(z)).view(-1, 32, 8, 8))
        y = y.clone()
        y[..., -1] = 0                                  # cot dem ky thuat = 0
        return y

    def forward(self, x):
        return self.decode(self.encode(x))


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        c = lambda i, o: nn.Conv2d(i, o, 3, 2, 1)
        self.f = nn.Sequential(c(1, 16), nn.ReLU(), c(16, 32), nn.ReLU(), c(32, 64), nn.ReLU(),
                               nn.Flatten(), nn.Dropout(0.2), nn.Linear(4096, 10))

    def forward(self, x):
        return self.f(x)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def rec_loss(xh, x, lam):
    e = (xh - x)[..., :T_VALID]                         # chi 63 khung hop le
    return (e ** 2).mean() + lam * e.abs().mean()


# ----------------------------------------------------------------------------- huan luyen
def train_ae(Xtr, Xva, seed, dz, lam, epochs, patience=20, bs=32):
    torch.manual_seed(seed)
    m = AE(dz).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed + 7)
    Xtr, Xva = Xtr.to(DEV), Xva.to(DEV)
    best, best_state, bad, hist = 1e18, None, 0, []
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), bs):
            xb = Xtr[perm[i:i + bs].to(DEV)]
            loss = rec_loss(m(xb), xb, lam)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            vl = rec_loss(m(Xva), Xva, lam).item()
        hist.append(vl)
        if vl < best:
            best, best_state, bad = vl, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    return m.eval(), dict(epochs_run=len(hist), best_val=best)


@torch.no_grad()
def predict(m, X):
    m.eval()
    out = [m(X[i:i + 500].to(DEV)).cpu() for i in range(0, len(X), 500)]
    return torch.cat(out)


def train_clf(Xtr, ytr, Xva, yva, seed, max_steps, every=100, patience=20, bs=32):
    torch.manual_seed(seed)                              # cung khoi tao cho moi dieu kien
    m = CNN().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)
    g = torch.Generator().manual_seed(seed + 1000)
    Xtr, ytr = Xtr.to(DEV), ytr.to(DEV)
    best_key, best_state, bad, step = (-1.0, 0.0), None, 0, 0
    t0 = time.time()
    while step < max_steps:
        m.train()
        idx = torch.randint(0, len(Xtr), (bs,), generator=g).to(DEV)
        loss = F.cross_entropy(m(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        step += 1
        if step % every == 0:
            lg = predict(m, Xva)
            acc = (lg.argmax(1) == yva).float().mean().item()
            ce = F.cross_entropy(lg, yva).item()
            key = (acc, -ce)                             # Accuracy cao nhat; hoa thi CE nho hon
            if key > best_key:
                best_key, best_state, bad = key, copy.deepcopy(m.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is None:
        best_state = m.state_dict()
    m.load_state_dict(best_state)
    return m.eval(), dict(steps_run=step, train_time=time.time() - t0, val_acc=best_key[0])


# ----------------------------------------------------------------------------- sinh du lieu (muc 3.4)
def slerp(z1, z2, a):
    r1 = z1.norm(dim=1, keepdim=True)
    r2 = z2.norm(dim=1, keepdim=True)
    u1, u2 = z1 / r1.clamp_min(1e-12), z2 / r2.clamp_min(1e-12)
    th = torch.acos((u1 * u2).sum(1, keepdim=True).clamp(-1, 1))
    sn = torch.sin(th)
    r = (1 - a) * r1 + a * r2
    zs = r * (torch.sin((1 - a) * th) / sn * u1 + torch.sin(a * th) / sn * u2)
    fallback = (r1 < 1e-6) | (r2 < 1e-6) | (sn.abs() < 1e-6)
    return torch.where(fallback, (1 - a) * z1 + a * z2, zs), int(fallback.sum())


def make_pairs(meta_tr, M, seed):
    """M/10 mau moi chu so; cac cap nguoi noi duoc lay deu; khong lap (file_1, file_2, alpha)."""
    rng = np.random.default_rng(seed)
    spk = sorted(meta_tr.speaker.unique())
    spk_pairs = list(combinations(spk, 2))
    pos = {(d, s): np.where((meta_tr.label.values == d) & (meta_tr.speaker.values == s))[0]
           for d in range(10) for s in spk}
    rows, seen = [], set()
    for d in range(10):
        order = []
        while len(order) < M // 10:
            order += list(rng.permutation(len(spk_pairs)))
        for k in order[:M // 10]:
            s1, s2 = spk_pairs[k]
            if rng.random() < 0.5:
                s1, s2 = s2, s1
            while True:
                i1, i2 = int(rng.choice(pos[(d, s1)])), int(rng.choice(pos[(d, s2)]))
                al = float(rng.choice(ALPHAS))
                if (i1, i2, al) not in seen:
                    seen.add((i1, i2, al)); break
            rows.append(dict(label=d, idx1=i1, idx2=i2, spk1=s1, spk2=s2, alpha=al,
                             file1=meta_tr.file.values[i1], file2=meta_tr.file.values[i2]))
    return pd.DataFrame(rows)


@torch.no_grad()
def generate(ae, Xtr, pairs):
    i1, i2 = torch.tensor(pairs.idx1.values), torch.tensor(pairs.idx2.values)
    a = torch.tensor(pairs.alpha.values, dtype=torch.float32).view(-1, 1).to(DEV)
    lin, sph, rec, fb = [], [], [], 0
    for s in range(0, len(pairs), 250):
        sl = slice(s, s + 250)
        x1, x2 = Xtr[i1[sl]].to(DEV), Xtr[i2[sl]].to(DEV)
        z1, z2 = ae.encode(x1), ae.encode(x2)
        lin.append(ae.decode((1 - a[sl]) * z1 + a[sl] * z2).cpu())
        zs, n = slerp(z1, z2, a[sl]); fb += n
        sph.append(ae.decode(zs).cpu())
        rec.append(ae.decode(z1).cpu())
    return torch.cat(lin), torch.cat(sph), torch.cat(rec), fb


def spec_mask(X, seed):
    """Mot mat na thoi gian + mot mat na tan so, do rong 0..8, dien 0 trong mien z-score."""
    rng = np.random.default_rng(seed)
    X = X.clone()
    for i in range(len(X)):
        wt = int(rng.integers(0, 9)); ts = int(rng.integers(0, T_VALID - wt + 1))
        wf = int(rng.integers(0, 9)); fs = int(rng.integers(0, N_MEL - wf + 1))
        X[i, 0, :, ts:ts + wt] = 0
        X[i, 0, fs:fs + wf, :T_VALID] = 0
    return X


def save_fig(path, x1, x2, rec, lin, sph, alphas_shown=(0.25, 0.5, 0.75)):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("X1 goc", x1), ("D(E(X1))", rec)]
    panels += [(f"Linear a={al}", lin[al]) for al in alphas_shown]
    panels += [("SLERP a=0.5", sph), ("X2 goc", x2)]
    fig, ax = plt.subplots(2, 4, figsize=(14, 6))
    for a_, (t, im) in zip(ax.ravel(), panels):
        h = a_.imshow(im[0, :, :T_VALID].numpy(), origin="lower", aspect="auto", vmin=-3, vmax=3,
                      cmap="magma")
        a_.set_title(t); a_.set_xlabel("khung"); a_.set_ylabel("mel-bin")
    fig.colorbar(h, ax=ax, shrink=0.8); fig.savefig(path, dpi=120); plt.close(fig)


# ----------------------------------------------------------------------------- thi nghiem TN1 + TN2
def cmd_run(a):
    out = a.out
    os.makedirs(f"{out}/ckpt", exist_ok=True); os.makedirs(f"{out}/pairs", exist_ok=True)
    meta = pd.read_csv(f"{out}/manifest.csv")
    X = np.load(f"{out}/features_raw.npy")
    splits = make_splits(meta)
    pd.DataFrame([dict(fold=s["fold"], test=s["test"], val=s["val"], train=",".join(s["train"]))
                  for s in splits]).to_csv(f"{out}/splits.csv", index=False)
    folds = a.folds or list(range(1, 7))
    seeds = a.seeds if a.seeds is not None else [0, 1, 2]
    ae_epochs, clf_steps, M = a.ae_epochs, a.clf_steps, a.M
    if a.quick:
        folds, seeds, ae_epochs, clf_steps, M = [1], [0], 3, 300, 100

    res_path, pred_path = f"{out}/results.csv", f"{out}/predictions.csv"
    done = set()
    if os.path.exists(res_path):
        d = pd.read_csv(res_path)
        done = set(zip(d.fold, d.seed))

    for sp in splits:
        if sp["fold"] not in folds:
            continue
        itr = np.where(meta.speaker.isin(sp["train"]))[0]
        iva = np.where(meta.speaker == sp["val"])[0]
        ite = np.where(meta.speaker == sp["test"])[0]
        mtr = meta.iloc[itr].reset_index(drop=True)
        mu, sd = X[itr].mean(axis=(0, 2)), X[itr].std(axis=(0, 2))   # chi tu Train cua fold
        np.savez(f"{out}/norm_fold{sp['fold']}.npz", mu=mu, sd=sd)
        Xtr, Xva, Xte = (norm_tensor(X[i], mu, sd) for i in (itr, iva, ite))
        ytr, yva, yte = (torch.tensor(meta.label.values[i]) for i in (itr, iva, ite))
        print(f"\n=== Fold {sp['fold']} | test={sp['test']} val={sp['val']} | "
              f"train {len(itr)} / val {len(iva)} / test {len(ite)}")
        assert not set(sp["train"]) & {sp["test"], sp["val"]} and sp["test"] != sp["val"]

        for seed in seeds:
            if (sp["fold"], seed) in done:
                print(f"  seed {seed}: da co ket qua, bo qua"); continue
            t0 = time.time()
            ae, aeinfo = train_ae(Xtr, Xva, seed, a.dz, a.lam, ae_epochs)
            torch.save(ae.state_dict(), f"{out}/ckpt/ae_f{sp['fold']}_s{seed}.pt")
            rec_te = predict(ae, Xte)
            e = (rec_te - Xte)[..., :T_VALID]
            mse, mae = (e ** 2).mean().item(), e.abs().mean().item()
            print(f"  seed {seed}: AE {n_params(ae):,} tham so, {aeinfo['epochs_run']} epoch, "
                  f"val_rec={aeinfo['best_val']:.4f} | test MSE={mse:.4f} MAE={mae:.4f}")

            pairs = make_pairs(mtr, M, seed=1000 * sp["fold"] + seed)
            pairs.to_csv(f"{out}/pairs/pairs_f{sp['fold']}_s{seed}.csv", index=False)
            assert not (set(pairs.spk1) | set(pairs.spk2)) & {sp["test"], sp["val"]}
            Xlin, Xsph, Xrec, nfb = generate(ae, Xtr, pairs)
            Xmask = spec_mask(Xtr[torch.tensor(pairs.idx1.values)], seed=seed + 55)
            yaug = torch.tensor(pairs.label.values)
            if sp["fold"] == 1 and seed == seeds[0]:      # hinh minh hoa TN2 (cung truc, cung thang mau)
                k0 = pairs.iloc[[0]]
                lin_k = {al: generate(ae, Xtr, k0.assign(alpha=al))[0][0] for al in ALPHAS}
                sph_k = generate(ae, Xtr, k0.assign(alpha=0.5))[1][0]
                save_fig(f"{out}/fig_interp_fold1.png", Xtr[pairs.idx1[0]], Xtr[pairs.idx2[0]],
                         Xrec[0], lin_k, sph_k)

            conds = {"B0_Real": None, "B1_SpecMask": Xmask, "B2_AE_Rec": Xrec,
                     "P1_AE_Linear": Xlin, "P2_AE_SLERP": Xsph}
            for name, aug in conds.items():
                Xc = Xtr if aug is None else torch.cat([Xtr, aug])
                yc = ytr if aug is None else torch.cat([ytr, yaug])
                clf, info = train_clf(Xc, yc, Xva, yva, seed, clf_steps)
                pr = predict(clf, Xte).argmax(1)
                acc = (pr == yte).float().mean().item()
                f1 = f1_score(yte.numpy(), pr.numpy(), average="macro", labels=list(range(10)),
                              zero_division=0)
                torch.save(clf.state_dict(), f"{out}/ckpt/clf_{name}_f{sp['fold']}_s{seed}.pt")
                row = dict(method=name, fold=sp["fold"], test_speaker=sp["test"], seed=seed,
                           M=0 if aug is None else M, d_z=a.dz, lam=a.lam, accuracy=acc,
                           macro_f1=f1, mse=mse if name.startswith(("B2", "P")) else np.nan,
                           mae=mae if name.startswith(("B2", "P")) else np.nan,
                           ae_params=n_params(ae), clf_params=n_params(clf),
                           steps_run=info["steps_run"], train_time_s=round(info["train_time"], 1),
                           slerp_fallbacks=nfb if name == "P2_AE_SLERP" else 0)
                pd.DataFrame([row]).to_csv(res_path, mode="a", header=not os.path.exists(res_path),
                                           index=False)
                pd.DataFrame(dict(method=name, fold=sp["fold"], seed=seed, file=meta.file.values[ite],
                                  y=yte.numpy(), pred=pr.numpy())).to_csv(
                    pred_path, mode="a", header=not os.path.exists(pred_path), index=False)
                print(f"    {name:<13} acc={acc:.4f} macroF1={f1:.4f} steps={info['steps_run']} "
                      f"({info['train_time']:.0f}s)")
            print(f"  seed {seed} xong sau {time.time() - t0:.0f}s | SLERP fallback: {nfb}/{M}")


# ----------------------------------------------------------------------------- tong hop (muc 4.5)
def cmd_aggregate(a):
    df = pd.read_csv(f"{a.out}/results.csv")
    # trung binh 3 seed cho tung nguoi noi, sau do mean +- SD (mau so 5) giua 6 nguoi noi
    per = df.groupby(["method", "test_speaker"])[["accuracy", "macro_f1"]].mean().reset_index()
    summ = per.groupby("method")[["accuracy", "macro_f1"]].agg(["mean", lambda s: s.std(ddof=1)])
    summ.columns = ["acc_mean", "acc_sd", "f1_mean", "f1_sd"]
    summ = summ.reset_index()
    print("\nBang 1. Ket qua tren nguoi noi chua thay (mean +- SD giua cac nguoi noi)")
    for _, r in summ.iterrows():
        print(f"  {r.method:<13} Acc {100 * r.acc_mean:6.2f} +- {100 * r.acc_sd:5.2f} | "
              f"MacroF1 {100 * r.f1_mean:6.2f} +- {100 * r.f1_sd:5.2f}")
    summ.to_csv(f"{a.out}/summary_methods.csv", index=False)

    spk_tab = per.pivot(index="test_speaker", columns="method", values="accuracy") * 100
    spk_tab.round(2).to_csv(f"{a.out}/summary_per_speaker.csv")
    print("\nBang 2. Accuracy (%) theo nguoi noi (TB 3 seed)\n", spk_tab.round(2).to_string())

    rows = []
    for base in ("B0_Real", "B1_SpecMask", "B2_AE_Rec"):
        for m in ("B1_SpecMask", "B2_AE_Rec", "P1_AE_Linear", "P2_AE_SLERP"):
            if m == base or m not in spk_tab or base not in spk_tab:
                continue
            d = spk_tab[m] - spk_tab[base]
            ps = df.pivot_table(index=["fold", "seed"], columns="method", values="accuracy")
            dseed = 100 * (ps[m] - ps[base])
            rows.append(dict(method=m, baseline=base, dAcc_mean_pp=d.mean(), dAcc_sd_pp=d.std(ddof=1),
                             speakers_improved=f"{int((d > 0).sum())}/{len(d)}",
                             runs_improved=f"{int((dseed > 0).sum())}/{len(dseed)}"))
    pd.DataFrame(rows).round(3).to_csv(f"{a.out}/summary_paired.csv", index=False)
    print("\nBang 3. Chenh lech ghep cap (diem phan tram)\n", pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nLuu y: mean +- SD khong tu chung minh y nghia thong ke; 18 ket qua KHONG phai 18 nguoi noi doc lap.")


# ----------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prep"); s.add_argument("--data", required=True); s.add_argument("--out", default="out")
    s = sub.add_parser("run"); s.add_argument("--out", default="out")
    s.add_argument("--folds", type=int, nargs="*"); s.add_argument("--seeds", type=int, nargs="*")
    s.add_argument("--M", type=int, default=1000); s.add_argument("--dz", type=int, default=64)
    s.add_argument("--lam", type=float, default=0.1)
    s.add_argument("--ae-epochs", type=int, default=200); s.add_argument("--clf-steps", type=int, default=6000)
    s.add_argument("--quick", action="store_true")
    s = sub.add_parser("aggregate"); s.add_argument("--out", default="out")
    a = p.parse_args()
    {"prep": cmd_prep, "run": cmd_run, "aggregate": cmd_aggregate}[a.cmd](a)
