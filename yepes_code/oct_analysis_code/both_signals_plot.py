import os
import csv
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import simpson
from matplotlib.offsetbox import AnchoredText
from matplotlib.lines import Line2D
from tqdm import tqdm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from class_area_functions import AreaDetector

import config
import area_testing

plt.style.use(hep.style.CMS)

SAVE_PLOTS = config.savePlot
SHOW_PLOTS = config.showPlot
OUT_DIR_OVERLAID = "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/generated_plots"

# ======================================================================
# USER PARAMETERS
# ======================================================================
INTEGRATION_METHOD = "adjusted"

HV_TARGET = 7.0

# IMPORTANT: you asked "no filter criteria" -> we ONLY filter by HV (and optional Z)
PLOT_ALL_Z = False          # set True to plot all Z
Z_FIXED    = 20.0          # used only when PLOT_ALL_Z = False

EXTENDED_CSV = (
    "/lustre/home/acota/medical_physics/output_flash_therapy_lgad_unison/data/"
    "lgad-2025-10-14_15/lgad-2025-10-14_15-clean_revisited.csv"
)

BASE_DIR = (
    "/lustre/home/acota/medical_physics/output_flash_therapy_lgad_unison/data/"
    "lgad-2025-10-14_15"
)

OUT_RESULTS_CSV = (
    "csvs_data/results/results_waveforms_window_integrals_HV-100_BEAM-85V.csv"
)

# If your scope files are split in these subfolders
SCOPE_SUBDIRS = ["2025-10-14", "2025-10-15"]

R_OHM = 50.0

def _safe_tag(s):
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    if s == "":
        s = "NA"
    return s

def _legend_block(time, sig, integ):
    t0w = float(integ.get("t0_win", np.nan))
    t1w = float(integ.get("t1_win", np.nan))

    Vmin_win, _ = vmin_tmin_in_window(time, sig, t0w, t1w)

    area = float(integ.get("area_win", np.nan))     # V·s
    q_c  = area / R_OHM                             # C
    q_nc = q_c * 1e9                                # nC

    # ---- FORMAT: X.XXX (3 decimals) for the 3 displayed values ----
    txt = (
        f"{'Vmin':<6} = {Vmin_win:.3f} V\n"
        f"{'Area':<6} = {area:.3e} V·s\n"
        f"{'Q':<6} = {q_nc:.3f} nC"
    )
    return txt

def vmin_tmin_in_window(time, sig, t0, t1):
    """
    Return (Vmin_window, tmin_window) computed ONLY from samples in [t0,t1].
    Fallback: global (Vmin,tmin) if window invalid.
    """
    time = np.asarray(time, float)
    sig  = np.asarray(sig, float)

    if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
        idx = int(np.nanargmin(sig))
        return float(sig[idx]), float(time[idx])

    mask = (time >= float(t0)) & (time <= float(t1))
    if np.sum(mask) < 3:
        idx = int(np.nanargmin(sig))
        return float(sig[idx]), float(time[idx])

    tw = time[mask]
    sw = sig[mask]
    idxw = int(np.nanargmin(sw))
    return float(sw[idxw]), float(tw[idxw])

def make_plot_filename(row, scope_basename, tag, ext="png"):
    z_val = float(row[0])
    hv_val = float(row[1])
    pw_val = float(row[2])
    beam = float(row[3])

    z_str  = _safe_tag(f"{z_val:.1f}".replace(".", "p"))
    hv_str = _safe_tag(f"{hv_val:.0f}")
    pw_str = _safe_tag(f"{pw_val:.3f}".replace(".", "p"))

    base = os.path.splitext(str(scope_basename))[0]
    base = _safe_tag(base)

    ext = ext.lstrip(".")  # allow "pdf" or ".pdf"
    return f"{tag}_Z{z_str}cm_HV{hv_str}V_PW{pw_str}us_BEAM{beam}_{base}.{ext}"

def plot_overlaid_one_axis(time,
                           corr_ch1, feats_ch1, integ_ch1, interp_ch1,
                           corr_ch2, feats_ch2, integ_ch2, interp_ch2,
                           row, scope_basename,
                           out_dir=OUT_DIR_OVERLAID,
                           save_plots=SAVE_PLOTS,
                           show_plots=SHOW_PLOTS):
    """
    NEW: Both waveforms in the SAME axes (Si Diode red, LGAD black).
    - Save in OUT_DIR_OVERLAID
    - Then (outside) the original separated plot is shown and saved in OUT_DIR_SEPARATED
    """
    print(integ_ch1)
    interp_ch2 = -0.41000000000000003
    t_us = np.asarray(time, float) * 1e6

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # store window positions (µs)
    vlines_ch1 = []
    vlines_ch2 = []

    # ============================================================
    # Si Diode (CH1)
    # ============================================================
    ax.plot(t_us, corr_ch1, color="red", lw=2, label="Si Diode")

    t0w1 = float(integ_ch1.get("t0_win", np.nan))
    t1w1 = float(integ_ch1.get("t1_win", np.nan))
    Vmin1, tmin1 = vmin_tmin_in_window(time, corr_ch1, t0w1, t1w1)
    ax.plot(tmin1 * 1e6, Vmin1, "o", color="red", ms=7)

    if np.isfinite(t0w1) and np.isfinite(t1w1) and t1w1 > t0w1:
        # STORE ONLY (NO ax.axvline)
        vlines_ch1.extend([t0w1 * 1e6, t1w1 * 1e6])

        mask1 = (time >= t0w1) & (time <= t1w1)
        y_fill1 = np.minimum(corr_ch1, 0.0)
        ax.fill_between(
            t_us[mask1],
            y_fill1[mask1],
            interp_ch1,
            color="red",
            alpha=0.25,
            zorder=1
        )

    # ============================================================
    # LGAD (CH2)
    # ============================================================
    ax.plot(t_us, corr_ch2, color="black", lw=2, label="LGAD")

    t0w2 = float(integ_ch2.get("t0_win", np.nan))
    t1w2 = float(integ_ch2.get("t1_win", np.nan))
    Vmin2, tmin2 = vmin_tmin_in_window(time, corr_ch2, t0w2, t1w2)
    ax.plot(tmin2 * 1e6, Vmin2, "o", color="black", ms=7)

    if np.isfinite(t0w2) and np.isfinite(t1w2) and t1w2 > t0w2:
        # STORE ONLY (NO ax.axvline)
        vlines_ch2.extend([t0w2 * 1e6, t1w2 * 1e6])

        mask2 = (time >= t0w2) & (time <= t1w2)
        y_fill2 = np.minimum(corr_ch2, 0.0)
        ax.fill_between(
            t_us[mask2],
            y_fill2[mask2],
            interp_ch2,
            color="black",
            alpha=0.20,
            zorder=1
        )

    spacer = Line2D([0], [0], color="none", lw=0, label="")

    handles = [
        Line2D([0], [0], color="red", lw=2, label="Si Diode"),
        Line2D([0], [0], color="black", lw=2, label="LGAD"),
    ]

 
    # ============================================================
    # Axes styling
    # ============================================================

    ax.axhline(0.0, color="blue", ls="--", lw=1.5, alpha=0.7)
    ax.axhline(interp_ch1, color="red", ls="--", lw=1.5, alpha=0.7)
    ax.axhline(interp_ch2, color="black", ls="--", lw=1.5, alpha=0.7)
    ax.set_xlabel("Time [µs]", fontsize=20)
    ax.set_ylabel("Voltage [V]", fontsize=20)
    ax.set_xlim(-5,5)
    ax.set_ylim(-1.8,0.5)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.legend(
        handles=handles,
        loc="best",
        ncol=1,
        prop={"family": "monospace", "size": 20},
        labelspacing=0.2,
        borderpad=1.2,
        handlelength=2.0,
        handletextpad=1.0,
        framealpha=0.95,
        frameon=False
    )
    print("PULSE SECTION")
    print(row)
 
    pw_val = float(row[2])
 
    info_text = (
        "• Fixed integration range method\n"
        + f"• Pulse duration {pw_val:.1f} " + r"$\mu s$"
    )
 
    ax.text(
        0.97, 0.48,
        info_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        multialignment="left",
        fontsize=14
    )
 
    if save_plots:
        for ext in ("png", "pdf"):
            out_name = make_plot_filename(row, scope_basename, tag="overlaid", ext=ext)
            out_path = os.path.join(out_dir, out_name)
            # dpi is mainly relevant for PNG; harmless for PDF (affects rasterized parts)
            fig.savefig(out_path, dpi=200 if ext == "png" else 300, bbox_inches="tight")
 
    if show_plots:
        plt.show()

    plt.close(fig)

def convert_times(time_step, start, index):
    return start+(index*time_step)

if __name__ == "__main__":
    '''return SignalResult(
            time=time,
            raw_signal=signal,
            corrected_signal=corrected_signal,
            interpolated_baseline=smoothed_baseline,
            area=pulse_area,
            indices={'start': start_idx, 'end': end_idx},
            metadata=params
        )'''
    date = "2025-10-15"
    Z = 20
    HV = 7
    pulse = 2.0
    fileno = "0775"
    print(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-{fileno}.csv")
    print(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-{fileno}.csv")

    range_dict = {}
    
    signal_area_list_diode = area_testing.signal_region_finder(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-{fileno}.csv", range_dict, f"{date}", pulse=2.0, Z=20, HV=7, beam="Electrons 85", ifile=1560, selected_channel="CH2", show_saturation_correction=False, mode='old')
    signal_area_list_lgad = area_testing.signal_region_finder(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-{fileno}.csv", range_dict, f"{date}", pulse=2.0, Z=20, HV=7, beam="Electrons 85", ifile=1560, selected_channel="CH1", show_saturation_correction=False, mode='old')

    
    [time_diode, signal_diode, corrected_signal_diode, remove_start_index_diode, remove_end_index_diode, shifted_start_index_diode, interpolated_baseline_diode, signal_area_diode, beam, pulse, Z, HV, file_path, date, baseline_times, baseline_signals] = signal_area_list_diode
    [time_lgad, signal_lgad, corrected_signal_lgad, remove_start_index_lgad, remove_end_index_lgad, shifted_start_index_lgad, interpolated_baseline_lgad, signal_area_lgad, beam, pulse, Z, HV, file_path, date, baseline_times, baseline_signals] = signal_area_list_lgad

    time_step = time_diode[1] - time_diode[0]
    remove_start_index_diode = convert_times(time_step, time_diode[0], remove_start_index_diode)
    remove_start_index_lgad = convert_times(time_step, time_lgad[0], remove_start_index_lgad)
    remove_end_index_diode = convert_times(time_step, time_diode[0], remove_end_index_diode)
    remove_end_index_lgad = convert_times(time_step, time_diode[0], remove_end_index_lgad)

    print(signal_lgad[0])
    if signal_lgad[0] > 0:
        print("true")
        shift = signal_lgad[0]
        signal_lgad = signal_lgad - shift

    area_args_diode = {'t0_full': 0, 't1_full': 0, 'tmin': time_diode[0], 't0_win': remove_start_index_diode, 't1_win': remove_end_index_diode, 'area_full': 0, 'area_win': signal_area_diode, 'pulse_us': pulse, 'manual': False}
    area_args_lgad = {'t0_full': 0, 't1_full': 0, 'tmin': time_lgad[0], 't0_win': remove_start_index_lgad, 't1_win': remove_end_index_lgad, 'area_full': 0, 'area_win': signal_area_lgad, 'pulse_us': pulse, 'manual': False}
    # test_area = {'t0_full': -4.115999999999998e-06, 't1_full': 2.2880000000000025e-06, 'tmin': -1.0019999999999982e-06, 't0_win': -1.4999999999999992e-06, 't1_win': -1.0019999999999982e-06, 'area_full': 1.9861102989661946e-06, 'area_win': 3.3217882928192336e-07, 'pulse_us': 0.5, 'manual': False}
    plot_overlaid_one_axis(time_lgad, signal_lgad, None, area_args_lgad, interpolated_baseline_lgad[shifted_start_index_lgad+1]-shift, signal_diode, None, area_args_diode, interpolated_baseline_diode[shifted_start_index_diode+1]-shift, [Z, HV, 2.0, 85], None, OUT_DIR_OVERLAID, SAVE_PLOTS, SHOW_PLOTS)