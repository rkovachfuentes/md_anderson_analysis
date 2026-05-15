#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ======================================================================
# FLASH RADIOTHERAPY — WAVEFORM A`<`zNALYZER (ADJUSTED ONLY)
#
# Adjusted integration rule (Pulse is in µs):
#   Pulse = 0.5  → integrate [tmin - 0.25, tmin + 0.25]  (µs)
#   Pulse = 1.0  → integrate [tmin - 0.50, tmin + 0.50]
#   Pulse = 2.0  → integrate [tmin - 1.00, tmin + 1.00]
#   Pulse = 3.0  → integrate [tmin - 1.50, tmin + 1.50]
#
# For asymmetric pulses:
#   1) Find the main negative bump as the contiguous region
#      with the largest |area| below a threshold.
#   2) The integration window has width = Pulse (in time),
#      is centered on tmin, and is then shifted/clamped so
#      that it stays inside the main bump as much as possible.
#   3) If the bump is narrower than Pulse, the window collapses
#      to the bump width.
#
# The adjusted area (Area_win) is used for charge, CSV and calibration.
# ======================================================================

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import simpson
from scipy.optimize import curve_fit
from tqdm import tqdm
import json
import re
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText

plt.style.use(hep.style.CMS)

# ======================================================================
# USER PARAMETERS
# ======================================================================
INTEGRATION_METHOD = "adjusted"

Z_TARGET    = 20.0
HV_TARGET   = 7.0
BEAM_TARGET = "Electrons 85V"

EXTENDED_CSV = (
    "/lustre/home/acota/medical_physics/output_flash_therapy_lgad_unison/data/lgad-2025-10-14_15/lgad-2025-10-15_14-clean.csv"
)

BASE_DIR = (
    "/lustre/home/acota/medical_physics/output_flash_therapy_lgad_unison/data/lgad-2025-10-14_15"
)

# Choose which channel you are processing in THIS run
# CHANNEL = "CH1"   # <-- change to "CH1" when you want channel_1 outputs
# CH_IDX = 1 if CHANNEL.upper() == "CH1" else 2





# Base output root
OUT_ROOT = (
    "/lustre/home/acota/medical_physics/Flash-therapy-LGAD-UNISON/yuca_jobs/run_waveform_processing/plots/all_individual_results"
)

def _slug(s: str) -> str:
    # "Electrons 85V" -> "Electrons85V" and remove weird chars
    s = str(s).strip().replace(" ", "")
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s


case_tag = f"Z{int(Z_TARGET)}_HV{int(HV_TARGET)}_{_slug(BEAM_TARGET)}"

CH_IDX = "dual"
CHANNEL = "CH1+CH2"

# Where the per-waveform table will be saved
OUT_CSV = os.path.join(
    OUT_ROOT,
    f"channel_{CH_IDX}",
    f"waveform_{case_tag}_{INTEGRATION_METHOD}_ch{CH_IDX}.csv"
)

OUT_PLOTS = os.path.join(
    OUT_ROOT,
    f"channel_{CH_IDX}",
    case_tag
)

os.makedirs(OUT_PLOTS, exist_ok=True)



MANUAL_LEFT_EXTEND_US  = 0.00   # extend left boundary earlier by this much
MANUAL_RIGHT_EXTEND_US = 0.00   # optional

# Map hardware channels to physical detector names
channel_label_map = {
    "CH1": "Detector: Si Diode",
    "CH2": "Detector: LGAD"
}

# ======================================================================
# IO + HELPERS
# ======================================================================
def read_scope_csv(path):
    """Read Tektronix CSV: returns time, CH1, CH2 as numpy arrays."""
    time, ch1, ch2 = [], [], []
    if not os.path.isfile(path):
        return None, None, None

    with open(path, "r") as f:
        reader = csv.reader(f)
        reading = False
        for row in reader:
            if not row:
                continue
            if row[0].strip().upper() == "TIME":
                reading = True
                continue
            if reading:
                try:
                    t  = float(row[0])
                    y1 = float(row[1]) if len(row) > 1 else np.nan
                    y2 = float(row[2]) if len(row) > 2 else np.nan
                    time.append(t)
                    ch1.append(y1)
                    ch2.append(y2)
                except Exception:
                    continue

    return np.array(time), np.array(ch1), np.array(ch2)


def baseline_and_correct(time, sig):
    """Linear baseline from first/last 15% of samples."""
    n = len(sig)
    if n < 10:
        return np.zeros_like(sig), sig.copy()

    n_pre  = max(5, int(0.15 * n))
    n_post = max(5, int(0.15 * n))

    pre   = np.mean(sig[:n_pre])
    post  = np.mean(sig[-n_post:])
    t_pre = np.mean(time[:n_pre])
    t_post= np.mean(time[-n_post:])

    slope = (post - pre) / (t_post - t_pre + 1e-18)

    if abs(slope) < 0.02 * abs(pre):
        baseline = np.interp(time, [t_pre, t_post], [pre, post])
    else:
        baseline = np.full_like(sig, pre)

    return baseline, sig - baseline



def _cumtrapz_increasing(x, f):
    """
    Cumulative trapezoid integral for increasing x.
    Returns A with A[0]=0 and A[k] = ∫_{x0}^{xk} f dx.
    """
    x = np.asarray(x, float)
    f = np.asarray(f, float)
    if len(x) < 2:
        return np.zeros_like(x)
    dx = np.diff(x)
    inc = 0.5 * (f[1:] + f[:-1]) * dx
    return np.concatenate([[0.0], np.cumsum(inc)])


def best_fixed_width_window_containing_tmin(time, y, a, b, tmin, w_target_s):
    """
    Choose [t0,t1] within [a,b] with width <= w_target_s that:
      - CONTAINS tmin
      - MAXIMIZES negative-only area (i.e. maximizes ∫ max(-y,0) dt)
    If the available negative lobe is narrower than w_target_s, returns the lobe bounds.
    """
    time = np.asarray(time, float)
    y    = np.asarray(y, float)

    mask = (time >= a) & (time <= b)
    if np.sum(mask) < 5:
        return float(a), float(b)

    t = time[mask]
    s = y[mask]

    # If the available region is already <= target width, just use it
    if (t[-1] - t[0]) <= w_target_s:
        return float(t[0]), float(t[-1])

    # negative-only integrand
    f = np.maximum(-s, 0.0)
    A = _cumtrapz_increasing(t, f)

    # index closest to tmin in this segment
    k0 = int(np.argmin(np.abs(t - tmin)))

    best_area = -1.0
    best_i = None
    best_j = None

    # left boundary candidates i must be <= k0 to ensure the window can contain tmin
    for i in range(0, k0 + 1):
        t_end = t[i] + w_target_s
        j = int(np.searchsorted(t, t_end, side="right") - 1)  # ensures width <= w_target_s
        if j <= i:
            continue
        if j < k0:  # window must contain tmin
            continue

        area = float(A[j] - A[i])
        if area > best_area:
            best_area = area
            best_i, best_j = i, j

    if best_i is None:
        # Fallback: clamp a window around tmin while respecting width and bounds
        # Try to start as late as possible but still include tmin and fit w_target_s.
        t0 = max(float(t[0]), float(t[k0] - w_target_s))
        t1 = min(float(t[-1]), t0 + w_target_s)
        return float(t0), float(t1)

    return float(t[best_i]), float(t[best_j])


def integrate_neg_only(time, y, a, b):
    """
    Integrate only the negative part of y in [a,b].
    Returns a positive number (area of negative lobe).
    """
    mask = (time >= a) & (time <= b) & (y < 0.0)
    if np.sum(mask) < 2:
        return 0.0
    return float(simpson(-y[mask], x=time[mask]))  # minus sign makes it positive



def choose_tmin_by_max_window_area(time, sig, pulse_us, sigma_base):
    """
    Choose tmin = the negative minimum that gives the largest *broad* negative area.
    This ignores very narrow or spiky transients (trigger artifacts).
    Works automatically even if the true pulse drifts in time.
    """
    time = np.asarray(time, float)
    sig  = np.asarray(sig, float)
    n = len(sig)
    if n < 20:
        return float(time[np.argmin(sig)])

    w = float(pulse_us) * 1e-6
    dt = np.median(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        return float(time[np.argmin(sig)])

    # smooth signal slightly for stability
    win = max(11, int(0.02 * w / dt))
    if win % 2 == 0:
        win += 1
    win = min(win, n - 1 if (n - 1) % 2 == 1 else n - 2)
    s_s = savgol_filter(sig, win, polyorder=3, mode="interp")

    # find all minima below baseline
    inv = -s_s
    peaks, _ = find_peaks(inv, height=4.0 * sigma_base)
    if len(peaks) == 0:
        return float(time[np.argmin(sig)])

    best_score, best_t = -1.0, float(time[np.argmin(sig)])

    for p in peaks:
        t_c = float(time[p])
        # integrate negative-only in ±(Pulse/2)
        t0 = t_c - 0.5 * w
        t1 = t_c + 0.5 * w
        A = integrate_neg_only(time, sig, t0, t1)
        # compute width above half depth to penalize narrow spikes
        half = 0.5 * sig[p]
        idxs = np.where(sig <= half)[0]
        if len(idxs) > 1:
            width = time[idxs[-1]] - time[idxs[0]]
        else:
            width = 0
        score = A * (width / (0.5 * w))  # weight by relative width
        if score > best_score:
            best_score, best_t = score, t_c

    return best_t



# ======================================================================
# FEATURE EXTRACTION
# ======================================================================
def extract_features(time, sig):
    """Basic waveform features from corrected signal sig (in mV)."""
    Vmin = np.min(sig)
    idxm = int(np.argmin(sig))
    tmin = time[idxm]

    # FWHM using half of Vmin
    half = Vmin / 2.0
    idxs = np.where(sig <= half)[0]
    FWHM = time[idxs[-1]] - time[idxs[0]] if len(idxs) > 1 else np.nan

    # Raw integrals over all time
    Q = float(simpson(-sig, x=time))       # mV·s
    E = float(simpson(sig**2, x=time))     # mV²·s

    # Baseline noise for SNR
    Nbase = max(10, int(0.10 * len(sig)))
    sigma_base = np.std(sig[:Nbase])
    SNR = abs(Vmin) / (sigma_base + 1e-18)

    # Peaks in inverted corrected signal
    inv = -sig
    peaks, _ = find_peaks(inv, height=0.10 * abs(Vmin))

    return dict(
        Vmin=Vmin,
        tmin=tmin,
        FWHM=FWHM,
        Q=Q,
        E=E,
        sigma_base=sigma_base,
        SNR=SNR,
        npeaks=len(peaks),
    )


# ======================================================================
# MAIN PULSE REGION + ADJUSTED INTEGRATION
# ======================================================================
from scipy.signal import find_peaks

def find_main_pulse_region(time, sig, pulse_us, sigma_base):
    """
    Find a reasonable [i0,i1] region for the main negative lobe.
    Uses find_peaks when possible; otherwise falls back to a threshold-based
    contiguous negative region around the global minimum (NOT idxm±1).
    """
    n = len(sig)
    if n < 10:
        idxm = int(np.argmin(sig))
        return max(idxm-1, 0), min(idxm+1, n-1), idxm

    time = np.asarray(time, float)
    sig  = np.asarray(sig, float)

    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        idxm = int(np.argmin(sig))
        return max(idxm-1, 0), min(idxm+1, n-1), idxm

    inv = -sig
    w_target = float(pulse_us) * 1e-6

    # IMPORTANT: allow much narrower peaks than 0.25*pulse
    min_width_s = max(30e-9, 0.05 * w_target)          # 5% of pulse (and >=30 ns)
    min_width_samples = max(int(min_width_s / dt), 5)

    prom = max(0.08 * float(np.nanmax(inv)), 6.0 * float(sigma_base))
    peaks, props = find_peaks(inv, prominence=prom, width=min_width_samples)

    # If we do find a peak, use its bases as before
    if len(peaks) > 0:
        k = int(np.argmax(props["prominences"]))
        p = int(peaks[k])
        i0 = int(props["left_bases"][k])
        i1 = int(props["right_bases"][k])
        i0 = max(0, min(i0, n-2))
        i1 = max(i0+1, min(i1, n-1))
        min_idx = i0 + int(np.argmin(sig[i0:i1+1]))
        return i0, i1, min_idx

    # --------- fallback: threshold-based contiguous negative region ----------
    idxm = int(np.argmin(sig))
    depth = abs(float(sig[idxm]))
    thr = -max(4.0 * float(sigma_base), 0.02 * depth)  # "meaningfully negative"

    is_neg = (sig < thr)

    # bridge short gaps (so you don’t split on tiny zero-crossings)
    gap_us = 0.08
    gap_samp = max(1, int((gap_us * 1e-6) / dt))
    is_neg2 = is_neg.copy()
    neg_idx = np.where(is_neg)[0]
    for a, b in zip(neg_idx[:-1], neg_idx[1:]):
        if 1 < (b - a) <= gap_samp:
            is_neg2[a:b+1] = True

    # find contiguous runs
    idx2 = np.where(is_neg2)[0]
    if len(idx2) == 0:
        return max(idxm-1, 0), min(idxm+1, n-1), idxm

    runs = []
    start = idx2[0]
    prev  = idx2[0]
    for k in idx2[1:]:
        if k == prev + 1:
            prev = k
        else:
            runs.append((start, prev))
            start = k
            prev  = k
    runs.append((start, prev))

    # pick run containing idxm (or closest)
    best = None
    for r0, r1 in runs:
        if r0 <= idxm <= r1:
            best = (r0, r1)
            break
    if best is None:
        dists = [min(abs(idxm-r0), abs(idxm-r1)) for r0, r1 in runs]
        best = runs[int(np.argmin(dists))]

    i0, i1 = best
    min_idx = i0 + int(np.argmin(sig[i0:i1+1]))
    return int(i0), int(i1), int(min_idx)


def red_area_bounds(time, y, i0, i1, min_idx, sigma_base):
    """
    Return tight [iL, iR] that matches the "red area":
    - integrate only negative
    - stop at the 'knee' on the right (slope change), not at right_bases

    Uses a smoothed derivative to detect the knee robustly.
    """
    # work only inside the detected main region
    t = time[i0:i1+1]
    s = y[i0:i1+1]
    kmin = min_idx - i0

    # --- smooth to make derivatives stable ---
    n = len(s)
    
    # window ~ 5% of segment, at least 11, must be odd
    win = max(11, int(0.1 * n))
    #win = 200
    
    if win % 2 == 0:
        win += 1
    if win >= n:
        win = n - 1 if (n - 1) % 2 == 1 else n - 2
        win = max(win, 5)

    s_s = savgol_filter(s, window_length=win, polyorder=3, mode="interp")

    # derivatives
    ds = np.gradient(s_s, t)   # first derivative
    # local minimum amplitude (negative)
    Vmin = float(s_s[kmin])

    # --- thresholds (robust) ---
    # amplitude threshold: stop once we've recovered to a small fraction of the peak depth
    AMP_FRAC = 0.20  # 0.10–0.30; smaller = tighter (less tail)
    amp_thr = -AMP_FRAC * abs(Vmin)

    # slope threshold: detect end of steep recovery
    # estimate "steepest recovery slope" after the min
    right_ds = ds[kmin:]
    if len(right_ds) < 3:
        return i0, i1

    ds_max = float(np.max(right_ds))  # should be positive
    SLOPE_FRAC = 0.35                 # 0.25–0.50; smaller = tighter
    slope_thr = SLOPE_FRAC * ds_max

    # also require we are safely above noise to avoid cutting too early
    eps = max(4.0 * float(sigma_base), 0.03 * abs(Vmin))

    # --- find right boundary (knee) ---
    # rule: first index after min where BOTH:
    #   (1) signal is close to baseline (>-eps) OR recovered to amp_thr
    #   (2) slope has dropped below slope_thr (end of steep part)
    iR = None
    for j in range(kmin, n):
        cond_amp = (s_s[j] >= amp_thr) or (s_s[j] >= -eps)
        cond_slope = (ds[j] <= slope_thr)
        if cond_amp and cond_slope:
            iR = j
            break
    if iR is None:
        # fallback: last negative sample before crossing -eps
        neg = np.where(s_s[kmin:] < -eps)[0]
        iR = kmin + int(neg[-1]) if len(neg) else min(kmin + 1, n - 1)

    # --- find left boundary (symmetric logic) ---
    left_ds = ds[:kmin+1]
    ds_min = float(np.min(left_ds))  # should be negative
    slope_thr_L = SLOPE_FRAC * ds_min  # negative threshold

    iL = None
    for j in range(kmin, -1, -1):
        cond_amp = (s_s[j] >= amp_thr) or (s_s[j] >= -eps)
        cond_slope = (ds[j] >= slope_thr_L)  # ds close to 0 from negative side
        if cond_amp and cond_slope:
            iL = j
            break
    if iL is None:
        neg = np.where(s_s[:kmin+1] < -eps)[0]
        iL = int(neg[0]) if len(neg) else max(kmin - 1, 0)

    # convert back to full-array indices
    return i0 + iL, i0 + iR


def optimize_peak_core_bounds(time, y, i0, i1, min_idx, pulse_us, sigma_base):
    """
    Dynamically choose [tL, tR] to integrate ONLY the main peak (core),
    by scanning constant-fraction depth thresholds and selecting the
    region whose boundaries coincide with inflection/knee points.

    Returns: tL, tR, area_neg, best_frac
    """
    t = time[i0:i1+1]
    s = y[i0:i1+1]
    kmin = min_idx - i0
    n = len(s)

    if n < 11:
        tL, tR = float(t[0]), float(t[-1])
        return tL, tR, integrate_neg_only(time, y, tL, tR), np.nan

    # sampling
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        tL, tR = float(t[0]), float(t[-1])
        return tL, tR, integrate_neg_only(time, y, tL, tR), np.nan

    # pulse target width in seconds
    w_target = float(pulse_us) * 1e-6

    # smoothing window tied to the pulse duration (stable derivatives)
    # aim ~ 5–15% of pulse in samples
    win = int(max(11, 0.10 * w_target / dt))
    #win = 200
    
    if win % 2 == 0:
        win += 1
    win = min(win, n-1 if (n-1) % 2 == 1 else n-2)
    win = max(win, 11)

    s_s = savgol_filter(s, window_length=win, polyorder=3, mode="interp")
    d1 = np.gradient(s_s, t)
    d2 = np.gradient(d1, t)

    Vmin = float(s_s[kmin])   # negative
    depth = abs(Vmin)

    # noise / baseline tolerance
    eps = max(6.0 * float(sigma_base), 0.02 * depth)

    # width constraints: allow shrinking vs pulse, but reject silly tiny windows
    w_min = 0.35 * w_target   # 0.25–0.50 typical
    w_max = 1.60 * w_target   # allow shoulder, but still reject long junk

    # scan depth fractions: larger frac => tighter core (more negative threshold)
    # start at 0.50 and go to 0.95: you want a "core", typically 0.75–0.95 works well
    fracs = np.linspace(0.20, 0.85, 30)

    best = None  # (score, tL, tR, area, frac)

    for frac in fracs:
        thr = frac * Vmin  # Vmin negative -> thr is less negative if frac<1

        # contiguous region around min where signal is <= thr (deep enough)
        L = kmin
        while L > 0 and s_s[L] <= thr:
            L -= 1
        if s_s[L] > thr:
            L += 1

        R = kmin
        while R < n-1 and s_s[R] <= thr:
            R += 1
        if s_s[R] > thr:
            R -= 1

        if R <= L:
            continue

        tL = float(t[L])
        tR = float(t[R])
        width = tR - tL
        if width <= 0:
            continue

        # optionally expand a tiny bit until near-baseline, BUT not into shallow structures
        # (keeps the core nature)
        # Here we only allow expansion while still "meaningfully negative"
        L2 = L
        while L2 > 0 and s_s[L2] < -eps and (t[kmin] - t[L2]) < 0.7 * w_target:
            if s_s[L2-1] > s_s[L2]:  # moving left increases towards baseline
                L2 -= 1
            else:
                break

        R2 = R
        while R2 < n-1 and s_s[R2] < -eps and (t[R2] - t[kmin]) < 0.7 * w_target:
            if s_s[R2+1] > s_s[R2]:  # moving right increases towards baseline
                R2 += 1
            else:
                break

        tL2 = float(t[L2])
        tR2 = float(t[R2])
        width2 = tR2 - tL2

        if width2 < w_min or width2 > w_max:
            continue

        # integrate negative-only using ORIGINAL corrected signal y (not smoothed)
        area = integrate_neg_only(time, y, tL2, tR2)
        if area <= 0:
            continue

        # curvature at boundaries ~ "knee"/inflection
        curv = abs(float(d2[L2])) + abs(float(d2[R2]))

        # penalize wide windows (we want core peak)
        score = (area * curv) / np.sqrt(width2 + 1e-18)

        # also penalize if the window includes too much shallow stuff:
        # compare median inside window to a deep reference
        inside_med = float(np.median(s_s[L2:R2+1]))
        if inside_med > 0.65 * Vmin:  # not deep enough overall -> likely includes shallow bump
            score *= 0.35

        if best is None or score > best[0]:
            best = (score, tL2, tR2, area, float(frac))

    if best is None:
        # fallback: just integrate negative part in [i0,i1]
        tL, tR = float(t[0]), float(t[-1])
        return tL, tR, integrate_neg_only(time, y, tL, tR), np.nan

    _, tL_best, tR_best, area_best, frac_best = best
    return tL_best, tR_best, area_best, frac_best


def _cumtrapz_outward(x, f):
    """
    Cumulative trapezoid integral that is monotonic even if x is decreasing.
    Returns array same length as x, with A[0]=0.
    """
    x = np.asarray(x, float)
    f = np.asarray(f, float)
    if len(x) < 2:
        return np.zeros_like(x)

    dx = np.abs(np.diff(x))  # <-- key: absolute spacing
    inc = 0.5 * (f[1:] + f[:-1]) * dx
    return np.concatenate([[0.0], np.cumsum(inc)])


def choose_asymmetric_bounds_by_area(time, y, a, b, tmin,
                                     frac_left=0.97, frac_right=0.985,
                                     sigma_base=0.0,
                                     w_target=None,
                                     wmin_frac=0.25, wmax_frac=1.60,
                                     left_pad_us=0.0, right_pad_us=0.0,
                                     neg_sigma_mult=2.0, neg_amp_frac=0.01):
    """
    Pick [tL,tR] inside [a,b] by capturing a fraction of negative area on each side of tmin.
    Then optionally pad left/right in time (ONLY while still meaningfully negative).

    IMPORTANT:
      - frac_left, frac_right must be in (0,1]. If you set >1 it will clamp to 1.
      - left_pad_us is the simplest knob to integrate more to the LEFT.
    """
    frac_left  = float(np.clip(frac_left,  1e-6, 1.0))
    frac_right = float(np.clip(frac_right, 1e-6, 1.0))

    mask = (time >= a) & (time <= b)
    if np.sum(mask) < 8:
        return float(a), float(b)

    t = np.asarray(time[mask], float)
    s = np.asarray(y[mask], float)

    neg = np.maximum(-s, 0.0)

    # index closest to tmin within this segment
    k0 = int(np.argmin(np.abs(t - tmin)))

    # left cumulative (moving outward to the left)
    tL = t[:k0+1][::-1]
    nL = neg[:k0+1][::-1]
    AL = _cumtrapz_outward(tL, nL)
    AL_tot = float(AL[-1])

    # right cumulative (moving outward to the right)
    tR = t[k0:]
    nR = neg[k0:]
    AR = _cumtrapz_outward(tR, nR)
    AR_tot = float(AR[-1])

    epsA = 1e-18
    if AL_tot < epsA and AR_tot < epsA:
        return float(a), float(b)

    # choose cut points by area fraction
    if AL_tot > epsA:
        targetL = frac_left * AL_tot
        iL = int(np.searchsorted(AL, targetL, side="left"))
        iL = min(iL, len(tL)-1)
        t_left = float(tL[iL])
    else:
        t_left = float(t[0])

    if AR_tot > epsA:
        targetR = frac_right * AR_tot
        iR = int(np.searchsorted(AR, targetR, side="left"))
        iR = min(iR, len(tR)-1)
        t_right = float(tR[iR])
    else:
        t_right = float(t[-1])

    # "meaningfully negative" threshold for padding
    Vmin_seg = float(np.min(s))
    depth = abs(Vmin_seg)
    eps_neg = max(neg_sigma_mult * float(sigma_base), neg_amp_frac * depth)

    # pad LEFT (only while still negative enough)
    if left_pad_us > 0.0:
        t_left_target = t_left - left_pad_us * 1e-6
        # move left boundary earlier but only within [a,b]
        t_left_target = max(float(a), t_left_target)
        # walk to the closest earlier sample and then extend while s < -eps_neg
        idx_left = np.where(t <= t_left_target)[0]
        if len(idx_left) > 0:
            j = int(idx_left[-1])
            while j > 0 and s[j] < -eps_neg:
                j -= 1
            # after loop, j is first "not negative"; start at j+1
            t_left = float(t[min(j+1, len(t)-1)])

    # pad RIGHT (only while still negative enough)
    if right_pad_us > 0.0:
        t_right_target = t_right + right_pad_us * 1e-6
        t_right_target = min(float(b), t_right_target)
        idx_right = np.where(t >= t_right_target)[0]
        if len(idx_right) > 0:
            j = int(idx_right[0])
            while j < len(t)-1 and s[j] < -eps_neg:
                j += 1
            t_right = float(t[max(j-1, 0)])

    # width clamps (recommended)
    if w_target is not None and w_target > 0:
        w = t_right - t_left
        wmin = wmin_frac * w_target
        wmax = wmax_frac * w_target

        if w < wmin:
            half = 0.5 * wmin
            t_left  = max(float(a),  tmin - half)
            t_right = min(float(b),  tmin + half)

        if w > wmax:
            # keep right-biased shrink: 45% left, 55% right
            left_keep  = 0.45 * wmax
            right_keep = 0.55 * wmax
            t_left  = max(float(a), tmin - left_keep)
            t_right = min(float(b), tmin + right_keep)

    return float(t_left), float(t_right)



def keep_negative_component_containing_tmin(time, y, tL, tR, tmin,
                                            sigma_base,
                                            neg_sigma_mult=2.0,
                                            neg_amp_frac=0.01,
                                            gap_us=0.08):
    """
    Inside [tL,tR], keep the negative run that contains tmin.
    Bridges short gaps (<= gap_us) where signal briefly rises above threshold.
    """
    time = np.asarray(time, float)
    y = np.asarray(y, float)

    mask = (time >= tL) & (time <= tR)
    if np.sum(mask) < 8:
        return float(tL), float(tR)

    t = time[mask]
    s = y[mask]

    # negativity threshold
    Vmin_seg = float(np.min(s))
    depth = abs(Vmin_seg)
    eps_neg = max(neg_sigma_mult * float(sigma_base), neg_amp_frac * depth)

    is_neg = (s < -eps_neg)

    # bridge short gaps in is_neg
    dt = float(np.median(np.diff(t)))
    gap_samp = max(1, int((gap_us * 1e-6) / max(dt, 1e-18)))

    idx = np.where(is_neg)[0]
    if len(idx) == 0:
        return float(tL), float(tR)

    # fill gaps between consecutive negative indices if the gap is small
    is_neg2 = is_neg.copy()
    for a_i, b_i in zip(idx[:-1], idx[1:]):
        if 1 < (b_i - a_i) <= gap_samp:
            is_neg2[a_i:b_i+1] = True

    # find contiguous runs
    idx2 = np.where(is_neg2)[0]
    runs = []
    start = idx2[0]
    prev = idx2[0]
    for k in idx2[1:]:
        if k == prev + 1:
            prev = k
        else:
            runs.append((start, prev))
            start = k
            prev = k
    runs.append((start, prev))

    # pick run that contains tmin (closest index)
    k0 = int(np.argmin(np.abs(t - tmin)))
    best = None
    for r0, r1 in runs:
        if r0 <= k0 <= r1:
            best = (r0, r1)
            break
    if best is None:
        # if tmin fell in a gap, choose the run closest to k0
        dists = [min(abs(k0-r0), abs(k0-r1)) for r0, r1 in runs]
        best = runs[int(np.argmin(dists))]

    r0, r1 = best
    return float(t[r0]), float(t[r1])

    
def process_adjusted_pulse(time, sig, pulse_us):
    """
    Strict FLASH-safe adjusted integration:

    1) Choose tmin as the location that MAXIMIZES the negative area
       inside a Pulse-wide window (rejects narrow spikes).
    2) Identify the main negative lobe containing tmin.
    3) Inside that lobe, select the sub-window of width <= Pulse
       that CONTAINS tmin and MAXIMIZES negative-only area.
    4) Integrate ONLY the negative part in that window.

    The integration window NEVER exceeds Pulse and NEVER leaves
    the physical signal shape.
    """
    corrected = np.asarray(sig, float)
    time = np.asarray(time, float)

    n = len(corrected)
    if n < 10 or len(time) != n:
        return dict(
            t0_full=np.nan, t1_full=np.nan, tmin=np.nan,
            t0_win=np.nan, t1_win=np.nan,
            area_full=0.0, area_win=0.0,
            pulse_us=float(pulse_us)
        )

    # ------------------------------------------------------------------
    # 0) Baseline noise estimate
    # ------------------------------------------------------------------
    Nbase = max(10, int(0.10 * n))
    sigma_base = float(np.std(corrected[:Nbase]))

    # ------------------------------------------------------------------
    # 1) Choose tmin by maximum negative-area in a Pulse-wide window
    # ------------------------------------------------------------------
    tmin = float(
        choose_tmin_by_max_window_area(
            time, corrected, pulse_us, sigma_base
        )
    )

    # ------------------------------------------------------------------
    # 2) Broad pulse region (FOR REPORTING ONLY)
    # ------------------------------------------------------------------
    i0, i1, _ = find_main_pulse_region(time, corrected, pulse_us, sigma_base)
    t0_full = float(time[i0])
    t1_full = float(time[i1])
    area_full = float(
        integrate_neg_only(time, corrected, t0_full, t1_full)
    )

    # ------------------------------------------------------------------
    # 3) Restrict to the negative lobe containing tmin
    # ------------------------------------------------------------------
    tL_lobe, tR_lobe = keep_negative_component_containing_tmin(
        time, corrected,
        t0_full, t1_full,
        tmin,
        sigma_base=sigma_base,
        neg_sigma_mult=2.0,
        neg_amp_frac=0.005,
        gap_us=0.08
    )

    # ------------------------------------------------------------------
    # 4) Inside that lobe, choose BEST window of width <= Pulse
    #    that CONTAINS tmin and MAXIMIZES negative area
    # ------------------------------------------------------------------
    w_target = float(pulse_us) * 1e-6  # seconds

    t0_win, t1_win = best_fixed_width_window_containing_tmin(
        time, corrected,
        tL_lobe, tR_lobe,
        tmin,
        w_target
    )

    # ------------------------------------------------------------------
    # 5) Final safety: enforce width <= Pulse (hard guarantee)
    # ------------------------------------------------------------------
    if (t1_win - t0_win) > w_target:
        if (tmin - t0_win) > (t1_win - tmin):
            t0_win = t1_win - w_target
        else:
            t1_win = t0_win + w_target

    # Clamp to acquisition limits
    t0_win = max(float(time.min()), float(t0_win))
    t1_win = min(float(time.max()), float(t1_win))

    # Pathological fallback (should be extremely rare)
    if t1_win <= t0_win:
        half = 0.5 * w_target
        t0_win = max(float(time.min()), tmin - half)
        t1_win = min(float(time.max()), tmin + half)

    # ------------------------------------------------------------------
    # 6) Final negative-only integration
    # ------------------------------------------------------------------
    area_win = float(
        integrate_neg_only(time, corrected, t0_win, t1_win)
    )

    

    return dict(
        t0_full=t0_full,
        t1_full=t1_full,
        tmin=tmin,
        t0_win=float(t0_win),
        t1_win=float(t1_win),
        area_full=area_full,
        area_win=area_win,
        pulse_us=float(pulse_us)
    )






# ======================================================================
# PLOTTING
# ======================================================================
def plot_waveform(time, original, baseline, corr,
                  feats, integ, ch_name, idx, row, fname):
    fname_clean = os.path.basename(fname).replace(".csv", "")
    t_us = time * 1e6

    fig, ax = plt.subplots(figsize=(12, 10))

    # waveforms
    #ax.plot(t_us, original, color="blue",  label=f"Original Waveform")
    #ax.plot(t_us, baseline, color="red",   label="Baseline")
    ax.plot(t_us, corr, color="green", label="Baseline-corrected waveform")

    # adjusted integration window (draw only if valid)
    t0w = float(integ.get("t0_win", np.nan))
    t1w = float(integ.get("t1_win", np.nan))
    if np.isfinite(t0w) and np.isfinite(t1w) and (t1w > t0w):
        x0 = t0w * 1e6
        x1 = t1w * 1e6
    
        ax.axvline(x0, color="orange", linestyle="--")
        ax.axvline(x1, color="orange", linestyle="--")
    
        # y-position for labels (a bit above bottom)
        y_text = ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    
        dx = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])  # horizontal offset
        
        ax.text(x0 - dx, y_text, r"$t_1$", color="orange",
                ha="right", va="bottom", fontsize=30, rotation=0)
        
        ax.text(x1 + dx, y_text, r"$t_2$", color="orange",
                ha="left", va="bottom", fontsize=30, rotation=0)


    # mark tmin
    ax.plot(feats["tmin"] * 1e6, feats["Vmin"], "ko", markersize=7)

    # shade adjusted window
    mask_adj = (time >= integ["t0_win"]) & (time <= integ["t1_win"])
    y_fill = np.minimum(corr, 0.0)  # clamp to <= 0 so only negative area is filled
    ax.fill_between(t_us[mask_adj], y_fill[mask_adj], 0.0, color="yellow", alpha=0.35)

    # text block
    text_block = (
        f"{'Vmin':<3} = {feats['Vmin']:.3f} mV\n"
        f"{'tmin':<3} = {feats['tmin']*1e6:.2f} us\n"
        f"{'FWHM':<3} = {feats['FWHM']*1e6:.2f} us\n"
        f"{'Charge':<3} = {(integ['area_win']*1e-3/50.0)*1e12:.2f} pC"
        #f"Q_raw(full) = {feats['Q']:.4e} mV·s\n"
        #f"Area_full   = {integ['area_full']:.4e} mV·s\n"
        #f"Area_win    = {integ['area_win']:.4e} mV·s\n"
        #f"E           = {feats['E']:.4e} mV²·s\n"
        #f"SNR         = {feats['SNR']:.2f}\n"
        #f"peaks       = {feats['npeaks']}"
    )
    ax.plot([], [], " ", label=text_block)

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    print(f"Idx={idx}")
    #ax.set_title(
    #    f"{ch_name} — adjusted integration (window width = Pulse)\n"
    #    f"Idx={idx} | File={fname_clean}\n"
    #    f"Z={row['Z']} cm | HV={row['HV']} V | "
    #    f"Beam={row['Beam']} | Pulse={row['Pulse']} µs",
    #    fontsize=22
    #)
    
    # Choose what will actually be printed
    DISPLAY_LABEL = channel_label_map.get(CHANNEL, CHANNEL)

    ax.text(
        0.0, 1.005, DISPLAY_LABEL,
        transform=ax.transAxes,
        fontsize=20, fontweight="bold",
        ha="left", va="bottom"
    )

    title_text = f"PW = {row['Pulse']} µs   |   Beam = {row['Beam']}"
    
    ax.text(
        0.465, 1.005, title_text,
        transform=ax.transAxes,
        fontsize=20, fontweight="bold",
        ha="left", va="bottom"
    )
        
    
    y_all = np.concatenate([original, corr])
    y_min, y_max = np.min(y_all), np.max(y_all)
    margin = max(0.1 * (y_max - y_min), 0.01)
    
    #user defined ylim range
    ax.set_ylim(y_min - margin, 0.5)

    
    # automatic ylim range
    # ax.set_ylim(y_min - margin, y_max + margin)


    # User defined xlim range
    ax.set_xlim(-5, 4) 

    # All range xlim
    #ax.set_xlim(t_us.min(), t_us.max())

    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Voltage [mV]")
    ax.legend(fontsize=22, loc="lower right", prop={"family": "monospace"})

    outname = f"{idx:05d}_{ch_name}_adjusted_{fname_clean}.png"
    plt.savefig(os.path.join(OUT_PLOTS, outname), dpi=150)
    plt.show()
    plt.close(fig)


def plot_waveform_dual(time,
                       corr1, feats1, integ1,
                       corr2, feats2, integ2,
                       idx, row, fname):

    t_us = time * 1e6
    fig, ax = plt.subplots(figsize=(13, 10))

    # --- waveforms (keep these as the ONLY legend entries) ---
    ax.plot(t_us, corr1, color="red",   lw=2, label="Si Diode")
    ax.plot(t_us, corr2, color="black", lw=2, label="LGAD")

    # --- minima ---
    ax.plot(feats1["tmin"] * 1e6, feats1["Vmin"], "o", color="red",   ms=7)
    ax.plot(feats2["tmin"] * 1e6, feats2["Vmin"], "o", color="black", ms=7)

    # --- integration windows ---
    for integ, col in [(integ1, "red"), (integ2, "black")]:
        t0 = float(integ.get("t0_win", float("nan")))
        t1 = float(integ.get("t1_win", float("nan")))
        if np.isfinite(t0) and np.isfinite(t1) and t1 > t0:
            ax.axvline(t0 * 1e6, color=col, ls="--", alpha=0.8)
            ax.axvline(t1 * 1e6, color=col, ls="--", alpha=0.8)

    mask1 = (time >= integ1["t0_win"]) & (time <= integ1["t1_win"])
    y1_fill = np.minimum(corr1, 0.0)
    ax.fill_between(t_us[mask1], y1_fill[mask1], 0.0,
                    color="red", alpha=0.25, zorder=1)


    mask2 = (time >= integ2["t0_win"]) & (time <= integ2["t1_win"])
    y2_fill = np.minimum(corr2, 0.0)
    ax.fill_between(t_us[mask2], y2_fill[mask2], 0.0,
                    color="black", alpha=0.20, zorder=1)

    integ_ref = integ1   # use Si Diode as reference

    # =========================================================
    # Single integration window (reference: Si Diode)
    # =========================================================
    t0w = float(integ1.get("t0_win", np.nan))
    t1w = float(integ1.get("t1_win", np.nan))
    
    if np.isfinite(t0w) and np.isfinite(t1w) and (t1w > t0w):
        x0 = t0w * 1e6
        x1 = t1w * 1e6
    
        ax.axvline(x0, color="red", linestyle="--", lw=2)
        ax.axvline(x1, color="red", linestyle="--", lw=2)
    
        # y-position for labels
        y_text = ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        dx = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    
        ax.text(x0 - dx, y_text, r"$t_1$", color="red",
                ha="right", va="bottom", fontsize=32)
    
        ax.text(x1 + dx, y_text, r"$t_2$", color="red",
                ha="left", va="bottom", fontsize=32)
    
    
    print(f"Idx={idx}")

    # --- zero line ---
    ax.axhline(0, color="gray", ls="--")

    # --- axes labels ---
    ax.set_xlabel("Time [µs]", size=30)
    ax.set_ylabel("Voltage [mV]", size=30)

    # --- your requested xlim/ylim ---
    y_all = np.concatenate([corr1, corr2])
    y_min, y_max = np.min(y_all), np.max(y_all)
    margin = max(0.1 * (y_max - y_min), 0.01)
    ax.set_ylim(y_min - margin, 0.5)

    # Automatic xlim range
    ax.set_xlim(-5, t_us.max())
    
    # user defined xlim range
    #ax.set_xlim(-5, 4)

    ax.tick_params(axis="both", which="major", labelsize=28)

    
    title_text = f"PW = {row['Pulse']} µs   |   Beam = {row['Beam']}"
    
    ax.text(
        0.34, 1.005, title_text,
        transform=ax.transAxes,
        fontsize=28, fontweight="bold",
        ha="left", va="bottom"
    )

    
    # --- compute Q in pC with the SAME conversion as single plot ---
    # area_win is in (mV * s). Convert mV->V (1e-3), divide by 50Ω, convert C->pC (1e12)
    Q1_pC = float(integ1["area_win"]) * 1e-3 / 50.0 * 1e12
    Q2_pC = float(integ2["area_win"]) * 1e-3 / 50.0 * 1e12

    # --- the info block in EXACT order (not part of legend) ---
    info = (
        "Si Diode\n"
        f"  Vmin = {feats1['Vmin']:.3f} mV\n"
        #f"  FWHM = {feats1['FWHM']*1e6:.2f} µs\n"
        f"  Q    = {Q1_pC:.2f} pC\n\n"
        "LGAD\n"
        f"  Vmin = {feats2['Vmin']:.3f} mV\n"
        #f"  FWHM = {feats2['FWHM']*1e6:.2f} µs\n"
        f"  Q    = {Q2_pC:.2f} pC"
    )

    # Put it inside the plot like a legend box, but fully controlled
    at = AnchoredText(info, loc="lower right", prop=dict(family="monospace", size=24),
                      frameon=True, borderpad=0.8)
    at.patch.set_facecolor("white")
    at.patch.set_alpha(0.9)
    at.patch.set_edgecolor("black")
    ax.add_artist(at)

    # Now the real legend only shows the 2 line labels
    ax.legend(loc="upper left", fontsize=24)

    plt.tight_layout()

    fname_clean = os.path.basename(fname).replace(".csv", "")
    outname = f"{idx:05d}_DUAL_{fname_clean}.png"
    outname = f"{idx:05d}_DUAL_{fname_clean}.pdf"
    plt.savefig(os.path.join(OUT_PLOTS, outname))
    
    plt.show()
    plt.close(fig)

# ======================================================================
# LOAD CSV AND SELECT CASE
# ======================================================================
df = pd.read_csv(EXTENDED_CSV)

df_case = df[
    (df["Z"] == Z_TARGET) &
    (df["HV"] == HV_TARGET) &
    (df["Beam"].astype(str).str.strip() == BEAM_TARGET)
].copy()

print(f"Total rows in CSV     : {len(df)}")
print(f"Rows for case Z={Z_TARGET}, HV={HV_TARGET}, Beam='{BEAM_TARGET}': "
      f"{len(df_case)}\n")

results = []
idx = 0

def make_entry(row, idx, ch_name, status, reason="", feats=None, integ=None):
    entry = dict(row)
    entry.update({
        "idx": idx,
        "channel": ch_name,
        "method": INTEGRATION_METHOD,
        "status": status,
        "reason": reason,
    })

    # default integ fields
    if integ is None:
        entry.update({
            "t0_full": np.nan, "t1_full": np.nan, "tmin_integ": np.nan,
            "t0_win": np.nan,  "t1_win": np.nan,
            "area_full": np.nan, "area_win": np.nan,
            "charge_C": np.nan,
        })
    else:
        entry.update({
            "t0_full": integ.get("t0_full", np.nan),
            "t1_full": integ.get("t1_full", np.nan),
            "tmin_integ": integ.get("tmin", np.nan),   # keep separate from feats["tmin"]
            "t0_win": integ.get("t0_win", np.nan),
            "t1_win": integ.get("t1_win", np.nan),
            "area_full": integ.get("area_full", np.nan),
            "area_win": integ.get("area_win", np.nan),
            "charge_C": (float(integ.get("area_win", np.nan)) / 50.0) if np.isfinite(integ.get("area_win", np.nan)) else np.nan,
        })

    # default feature fields (keep same keys every row)
    if feats is None:
        entry.update({
            "Vmin": np.nan, "tmin": np.nan, "FWHM": np.nan,
            "Q": np.nan, "E": np.nan,
            "sigma_base": np.nan, "SNR": np.nan, "npeaks": np.nan,
        })
    else:
        entry.update(feats)

    return entry



# ======================================================================
# MAIN LOOP (PLOT ORDER: 0.5 -> 1.0 -> 2.0 -> 3.0)
# ======================================================================
'''results = []
idx = 0

print("Beginning waveform processing for selected case...\n")

df_case = df_case.copy()
df_case["Pulse"] = pd.to_numeric(df_case["Pulse"], errors="coerce")

PULSE_ORDER = [0.5, 1.0, 2.0, 3.0]

for pulse_target in PULSE_ORDER:
    df_p = df_case[np.isclose(df_case["Pulse"].to_numpy(dtype=float), pulse_target, rtol=0.0, atol=1e-12)]

    print(f"\n--- Plotting Pulse = {pulse_target} µs | N = {len(df_p)} ---")

    for _, row in tqdm(df_p.iterrows(), total=len(df_p), desc=f"Pulse={pulse_target} µs"):

        ch_name = "CH1+CH2"
        fname = str(row.get("FileName", "")).strip()

        # --- 1) Find scope file ---
        fpath = None
        for d in ["2025-10-14", "2025-10-15"]:
            test = os.path.join(BASE_DIR, d, fname)
            if os.path.isfile(test):
                fpath = test
                break

        if fpath is None:
            results.append(make_entry(row, idx, ch_name, status="missing_file", reason=f"File not found: {fname}"))
            idx += 1
            continue

        # --- 2) Read scope CSV ---
        time, ch1, ch2 = read_scope_csv(fpath)
        if time is None or len(time) == 0:
            results.append(make_entry(row, idx, ch_name, status="empty_waveform", reason=f"Empty read: {fname}"))
            idx += 1
            continue


        # =========================================================
        # ===============   CH1  (Si Diode)   =====================
        # =========================================================
        baseline1, corr1 = baseline_and_correct(time, ch1)
        feats1 = extract_features(time, corr1)

        V_raw = ch1
        V_range = float(np.nanmax(V_raw) - np.nanmin(V_raw))
        V_std   = float(np.nanstd(V_raw))
        TV      = float(np.nansum(np.abs(np.diff(V_raw))))

        feats1["V_range"] = V_range
        feats1["V_std"]   = V_std
        feats1["TV"]      = TV

        is_horizontal_1 = (
            (V_range < 6.0 * feats1["sigma_base"]) and
            (V_std   < 3.0 * feats1["sigma_base"]) and
            (TV      < 10.0 * feats1["sigma_base"])
        )


        # =========================================================
        # ===============   CH2  (LGAD)       =====================
        # =========================================================
        baseline2, corr2 = baseline_and_correct(time, ch2)
        feats2 = extract_features(time, corr2)

        V_raw = ch2
        V_range = float(np.nanmax(V_raw) - np.nanmin(V_raw))
        V_std   = float(np.nanstd(V_raw))
        TV      = float(np.nansum(np.abs(np.diff(V_raw))))

        feats2["V_range"] = V_range
        feats2["V_std"]   = V_std
        feats2["TV"]      = TV

        is_horizontal_2 = (
            (V_range < 6.0 * feats2["sigma_base"]) and
            (V_std   < 3.0 * feats2["sigma_base"]) and
            (TV      < 10.0 * feats2["sigma_base"])
        )


        # ---------------------------
        # HARD REJECT: wrong scope settings / no pulse
        # ---------------------------
        span_us = float((np.nanmax(time) - np.nanmin(time)) * 1e6)
        dt_ns   = float(np.nanmedian(np.diff(time)) * 1e9) if len(time) > 2 else np.nan

        if (not np.isfinite(span_us)) or (span_us < 2.0) or (span_us > 20.0):
            results.append(make_entry(row, idx, ch_name, status="reject_bad_timespan",
                                      reason=f"span_us={span_us:.3g}", feats=feats1))
            idx += 1
            continue

        if (not np.isfinite(dt_ns)) or (dt_ns < 0.2) or (dt_ns > 10.0):
            results.append(make_entry(row, idx, ch_name, status="reject_bad_dt",
                                      reason=f"dt_ns={dt_ns:.3g}", feats=feats1))
            idx += 1
            continue


        # -------------------------------------------------
        # Reject ONLY truly horizontal noise
        # -------------------------------------------------
        if is_horizontal_1 and is_horizontal_2:
            results.append(
                make_entry(
                    row, idx, ch_name,
                    status="reject_horizontal_noise",
                    reason="Both channels flat",
                    feats=feats1
                )
            )
            idx += 1
            continue


        # --- 4) Pulse must be valid ---
        pulse_val = row.get("Pulse", np.nan)
        if not np.isfinite(pulse_val):
            results.append(make_entry(row, idx, ch_name, status="invalid_pulse", reason="Pulse is NaN", feats=feats1))
            idx += 1
            continue

        pulse_us = float(pulse_val)


        # =========================================================
        # ==================  INTEGRATION  =======================
        # =========================================================
        integ1 = process_adjusted_pulse(time, corr1, pulse_us)
        integ2 = process_adjusted_pulse(time, corr2, pulse_us)


        # --- 8) Plot dual detector figure ---
        plot_waveform_dual(
            time,
            corr1, feats1, integ1,
            corr2, feats2, integ2,
            idx, row, fname
        )


        # --- 9) Store (keep CH1 as reference channel in CSV) ---
        results.append(make_entry(row, idx, "CH1", status="ok", feats=feats1, integ=integ1))
        results.append(make_entry(row, idx, "CH2", status="ok", feats=feats2, integ=integ2))

        idx += 1


# ======================================================================
# SAVE PER-WAVEFORM RESULTS
# ======================================================================
df_out = pd.DataFrame(results)

# 1) Save the full per-waveform table (this is your main output)
df_out.to_csv(OUT_CSV, index=False)

# 2) (Optional alias) Save points file for downstream plotting
points_out = os.path.join(OUT_PLOTS, "charge_vs_dose_points.csv")
df_out.to_csv(points_out, index=False)

print(f"\nWaveform processing finished.")
print(f"Plots saved in: {OUT_PLOTS}")
print(f"Waveform CSV saved: {OUT_CSV}\n")

# ======================================================================
# CHARGE vs DOSE CALIBRATION (USING ADJUSTED CHARGE)
# ======================================================================
if len(df_out) == 0:
    print("No events for the selected case. Skipping calibration plots.")
else:
    df_out = df_out.copy()
    df_out["charge_uC"] = df_out["charge_C"] * 1e6

    # >>> ADD THESE LINES RIGHT HERE <<<
    df_use = df_out[df_out["status"] == "ok"].copy()

    if len(df_use) == 0:
        print("No OK events after filtering (status=='ok'). Skipping calibration plots.")
        raise SystemExit(0)   # or just `pass` if you prefer

    # ---------------------------
    # Save all per-shot points (optional alias; df_out is already your master table)
    # ---------------------------
    points_out = os.path.join(OUT_PLOTS, "charge_vs_dose_points.csv")
    df_out.to_csv(points_out, index=False)

    # ---------------------------
    # Group by dose (binned means)
    # ---------------------------
    group = df_use.groupby("Dose", sort=True)
    dose_vals   = group["Dose"].first().to_numpy(dtype=float)
    charge_mean = group["charge_uC"].mean().to_numpy(dtype=float)
    charge_std  = group["charge_uC"].std(ddof=1).to_numpy(dtype=float)

    # protect against NaN/zero uncertainties
    charge_std = np.nan_to_num(charge_std, nan=1e-6)
    charge_std[charge_std <= 0.0] = 1e-6

    # save binned table
    df_binned = pd.DataFrame({
        "Dose": dose_vals,
        "charge_mean_uC": charge_mean,
        "charge_std_uC": charge_std,
        "N_events": group.size().to_numpy(dtype=int),
    })
    binned_out = os.path.join(OUT_PLOTS, "charge_vs_dose_binned.csv")
    df_binned.to_csv(binned_out, index=False)

    # ---------------------------
    # Linear fit
    # ---------------------------
    def model(D, alpha, beta):
        return alpha + beta * D

    popt, pcov = curve_fit(
        model,
        dose_vals,
        charge_mean,
        sigma=charge_std,
        absolute_sigma=True
    )

    alpha, beta = popt

    # choose 1σ or 2σ consistently
    SIGMA_FACTOR = 2.0  # keep 2.0 if you want 2σ; set 1.0 for 1σ
    alpha_err, beta_err = SIGMA_FACTOR * np.sqrt(np.diag(pcov))

    fit_vals = model(dose_vals, alpha, beta)

    # ---------------------------
    # Metrics
    # ---------------------------
    residuals = charge_mean - fit_vals
    chi2 = float(np.sum((residuals / charge_std) ** 2))
    ndof = int(len(dose_vals) - 2)
    chi2_ndof = float(chi2 / ndof) if ndof > 0 else np.nan

    SS_tot = float(np.sum((charge_mean - np.mean(charge_mean)) ** 2))
    SS_res = float(np.sum((charge_mean - fit_vals) ** 2))
    R2 = float(1.0 - SS_res / SS_tot) if SS_tot > 0 else np.nan

    denom = np.where(charge_mean != 0.0, charge_mean, np.nan)
    residuals_pct     = (residuals / denom) * 100.0
    residuals_pct_err = (charge_std / denom) * 100.0
    MAPR = float(np.nanmean(np.abs(residuals_pct)))

    # ---------------------------
    # Smooth curve + band (compute band BEFORE saving)
    # ---------------------------
    D_smooth   = np.linspace(0.0, float(np.max(dose_vals)), 400)
    fit_smooth = model(D_smooth, alpha, beta)

    J = np.vstack([np.ones_like(D_smooth), D_smooth]).T
    var_fit  = np.einsum("ij,jk,ik->i", J, pcov, J)
    fit_band = SIGMA_FACTOR * np.sqrt(var_fit)

    # save fit curve used for plotting
    df_curve = pd.DataFrame({
        "Dose_smooth": D_smooth,
        "fit_smooth_uC": fit_smooth,
        "fit_band_uC": fit_band
    })
    curve_out = os.path.join(OUT_PLOTS, "charge_vs_dose_fit_curve.csv")
    df_curve.to_csv(curve_out, index=False)

    # save fit params + metrics (everything needed to reproduce the figure text)
    fit_dict = {
        "model": "Q = alpha + beta * Dose",
        "sigma_factor": float(SIGMA_FACTOR),
        "alpha_uC": float(alpha),
        "alpha_err_uC": float(alpha_err),
        "beta_uC_per_Gy": float(beta),
        "beta_err_uC_per_Gy": float(beta_err),
        "covariance_matrix": pcov.tolist(),
        "chi2": float(chi2),
        "ndof": int(ndof),
        "chi2_ndof": float(chi2_ndof),
        "R2": float(R2),
        "MAPR_percent": float(MAPR),
        "Z_cm": float(Z_TARGET),
        "HV_V": float(HV_TARGET),
        "Beam": str(BEAM_TARGET),
        "integration_method": str(INTEGRATION_METHOD),
        "points_csv": os.path.basename(points_out),
        "binned_csv": os.path.basename(binned_out),
        "curve_csv": os.path.basename(curve_out)
    }
    fit_json = os.path.join(OUT_PLOTS, "charge_vs_dose_fit.json")
    with open(fit_json, "w") as f:
        json.dump(fit_dict, f, indent=4)

    # ======================================================================
    # ADDED: SAVE ALL LINEAR-FIT INFO (CHARGE vs DOSE) INTO A SINGLE CSV
    # (Only addition; no other behavior changed)
    # ======================================================================
    df_fit_all = pd.DataFrame({
        "Dose": dose_vals,
        "charge_mean_uC": charge_mean,
        "charge_std_uC": charge_std,
        "fit_uC": fit_vals,
        "residual_uC": residuals,
        "residual_pct": residuals_pct,
        "residual_pct_err": residuals_pct_err,
        "N_events": group.size().to_numpy(dtype=int),
        "alpha_uC": np.full_like(dose_vals, float(alpha), dtype=float),
        "alpha_err_uC": np.full_like(dose_vals, float(alpha_err), dtype=float),
        "beta_uC_per_Gy": np.full_like(dose_vals, float(beta), dtype=float),
        "beta_err_uC_per_Gy": np.full_like(dose_vals, float(beta_err), dtype=float),
        "chi2": np.full_like(dose_vals, float(chi2), dtype=float),
        "ndof": np.full_like(dose_vals, int(ndof), dtype=int),
        "chi2_ndof": np.full_like(dose_vals, float(chi2_ndof), dtype=float),
        "R2": np.full_like(dose_vals, float(R2), dtype=float),
        "MAPR_percent": np.full_like(dose_vals, float(MAPR), dtype=float),
        "sigma_factor": np.full_like(dose_vals, float(SIGMA_FACTOR), dtype=float),
        "pcov_00": np.full_like(dose_vals, float(pcov[0, 0]), dtype=float),
        "pcov_01": np.full_like(dose_vals, float(pcov[0, 1]), dtype=float),
        "pcov_10": np.full_like(dose_vals, float(pcov[1, 0]), dtype=float),
        "pcov_11": np.full_like(dose_vals, float(pcov[1, 1]), dtype=float),
    })
    fit_all_out = os.path.join(OUT_PLOTS, "charge_vs_dose_linear_fit_all_info.csv")
    df_fit_all.to_csv(fit_all_out, index=False)

    # ---------------------------
    # Plots
    # ---------------------------
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 12), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # 1) plot ALL individual points (per shot) with idx labels
    for D, sub in df_use.groupby("Dose", sort=True):
        x = np.full(len(sub), float(D))
        y = sub["charge_uC"].to_numpy()

        ax_top.scatter(x, y, s=35, color="gray", alpha=0.55, zorder=2)

        for xi, yi, idx_i in zip(x, y, sub["idx"].astype(int).to_numpy()):
            ax_top.annotate(
                str(idx_i),
                (xi, yi),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left", va="center",
                fontsize=9, color="gray",
                alpha=0.9, zorder=3
            )

    # 2) mean + intervals in black
    ax_top.errorbar(
        dose_vals, charge_mean, yerr=charge_std,
        fmt="o", color="black", capsize=4, zorder=4,
        label="Mean charge (adjusted window)"
    )

    ax_top.plot(
        D_smooth, fit_smooth, color="blue", linewidth=2,
        label="Fit: Q = α + β·D", zorder=3
    )

    band_label = f"{int(SIGMA_FACTOR)}σ fit band" if SIGMA_FACTOR in (1.0, 2.0) else f"{SIGMA_FACTOR}×σ fit band"
    ax_top.fill_between(
        D_smooth,
        fit_smooth - fit_band,
        fit_smooth + fit_band,
        color="blue", alpha=0.25, label=band_label, zorder=1
    )

    ax_top.set_ylabel("Charge (µC)", fontsize=18)
    ax_top.set_xlim(left=0)
    ax_top.set_title(
        f"{CHANNEL} Charge vs Dose (adjusted window, width = Pulse)\n"
        f"Z={Z_TARGET} cm, HV={HV_TARGET} V, Beam={BEAM_TARGET}",
        fontsize=22
     )
    #ax_top.grid(alpha=0.3)
    


    
    fit_text = (
        f"α = {alpha:.4f} ± {alpha_err:.4f} µC\n"
        f"β = {beta:.4f} ± {beta_err:.4f} µC/Gy\n"
        f"χ²/ndf = {chi2_ndof:.3f}\n"
        f"R² = {R2:.4f}\n"
        f"MAPR = {MAPR:.2f}%"
    )
    ax_top.text(
        0.30, 0.80, fit_text,
        transform=ax_top.transAxes,
        fontsize=14, va="top", ha="right",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
    )
    ax_top.legend(fontsize=14, loc="upper left")

    # bottom panel
    ax_bot.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_bot.errorbar(
        dose_vals, residuals_pct, yerr=residuals_pct_err,
        fmt="o", color="red", capsize=4, label="Residuals (%)"
    )
    ax_bot.set_xlabel("Dose (Gy)", fontsize=18)
    ax_bot.set_ylabel("Residuals (%)", fontsize=16)
    ax_bot.set_xlim(left=0)
    ax_bot.grid(alpha=0.3)
    ax_bot.legend(fontsize=12, loc="upper left")

    plt.tight_layout()
    calib_plot = os.path.join(
        OUT_PLOTS,
        "charge_vs_dose_fit_and_residuals_pct_adjusted_window_pulse.png"
    )
    plt.savefig(calib_plot, dpi=150)
    plt.show()
    plt.close(fig)

    print("\nFit results using adjusted-window charge:")
    print(f"  alpha = {alpha:.6f} ± {alpha_err:.6f} µC")
    print(f"  beta  = {beta:.6f} ± {beta_err:.6f} µC/Gy")
    print(f"  chi2/ndf = {chi2_ndof:.4f}")
    print(f"  R² = {R2:.6f}")
    print(f"  MAPR = {MAPR:.3f}%")
    print(f"Saved:")
    print(f"  points : {points_out}")
    print(f"  binned : {binned_out}")
    print(f"  curve  : {curve_out}")
    print(f"  fit    : {fit_json}")
    print(f"  fitcsv : {fit_all_out}")
    print(f"  plot   : {calib_plot}\n")
'''
