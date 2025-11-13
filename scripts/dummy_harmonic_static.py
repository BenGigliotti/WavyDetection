# static_tubing_sim_to_excel.py

import numpy as np
import pandas as pd
from datetime import datetime, timezone

def build_speed_profile(fs, segments):
    """
    segments: list of (duration_seconds, speed_fpm)
    Returns:
        t: time array [s]
        speed_fpm: speed array [fpm], length = sum(duration) * fs
    """
    dt = 1.0 / fs
    speeds = []
    for duration, speed in segments:
        n = int(duration * fs)
        speeds.append(np.full(n, speed, dtype=float))
    speed_fpm = np.concatenate(speeds)
    N = len(speed_fpm)
    t = np.arange(N) * dt
    return t, speed_fpm

def simulate_od_from_speed(
    fs,
    t,
    speed_fpm,
    mean_od_mm=12.7,
    noise_std_running=0.005,   # noise when line is moving [mm]
    noise_std_stopped=0.0005,  # noise when line is stopped [mm]
    drift_per_ft_mm=0.0,       # slow drift per foot of tube [mm/ft]
    chatter_wavelengths_in=None,  # list of spatial wavelengths [in]
    chatter_amps_mm=None          # list of amplitudes [mm]
):
    """
    Simulate OD measurement given time, speed profile, and spatial chatter info.
    """
    if chatter_wavelengths_in is None:
        chatter_wavelengths_in = []
    if chatter_amps_mm is None:
        chatter_amps_mm = [0.0] * len(chatter_wavelengths_in)

    assert len(chatter_wavelengths_in) == len(chatter_amps_mm), \
        "chatter_wavelengths_in and chatter_amps_mm must be same length"

    N = len(t)
    dt = 1.0 / fs

    # Convert speed to inches per second for spatial → temporal mapping
    speed_in_s = speed_fpm * (12.0 / 60.0)  # 1 fpm = 0.2 in/s

    # Cumulative length in feet for drift term
    # length_in = ∫ v_in_s dt, then convert to ft
    length_in = np.cumsum(speed_in_s) * dt
    length_ft = length_in / 12.0

    # Start OD as mean + drift along length
    od_mm = mean_od_mm + drift_per_ft_mm * length_ft

    # Add chatter components using phase accumulation
    for lam_in, amp_mm in zip(chatter_wavelengths_in, chatter_amps_mm):
        # instantaneous frequency f(t) = v_in_s / λ
        freq_t = np.where(lam_in > 0, speed_in_s / lam_in, 0.0)
        # phase[n] = phase[n-1] + 2π * f[n] / fs
        phase = 2 * np.pi * np.cumsum(freq_t) / fs
        od_mm += amp_mm * np.sin(phase)

    # Add noise: lower when speed=0
    moving_mask = speed_fpm > 0.0
    noise = np.zeros(N)
    noise[moving_mask] = np.random.normal(0.0, noise_std_running, moving_mask.sum())
    noise[~moving_mask] = np.random.normal(0.0, noise_std_stopped, (~moving_mask).sum())
    od_mm += noise

    # Optionally force OD to be nearly constant when speed == 0
    # (here we already used tiny noise_std_stopped, so it's almost constant)
    return od_mm

def make_timestamp_series(start_time, N, fs):
    """
    Create a pandas Series of timestamps starting at start_time with spacing 1/fs.
    """
    dt_s = 1.0 / fs
    offsets = pd.to_timedelta(np.arange(N) * dt_s, unit="s")
    return start_time + offsets

def main():
    # ---------------- CONFIG ----------------
    fs = 2400.0            # Hz
    # speed segments: (duration [s], speed [fpm])
    segments = [
        (10.0, 0.0),       # 10 s stopped
        (60.0, 150.0),     # 60 s at 150 fpm
        (10.0, 0.0),       # 10 s stopped
        (60.0, 200.0),     # 60 s at 200 fpm
    ]

    mean_od_mm = 12.7      # product target OD
    drift_per_ft_mm = 0.00001  # OD increases 0.001 mm per foot (example)

    # Spatial chatter patterns along the tube:
    # e.g. 0.5 inch and 2 inch wavelengths
    chatter_wavelengths_in = [0.5, 2.0]
    chatter_amps_mm = [0.05, 0.02]  # amplitude in mm

    noise_std_running = 0.005
    noise_std_stopped = 0.0005

    excel_filename = f'dummy_freq{fs}_mean{mean_od_mm}_drift{drift_per_ft_mm}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.xlsx'

    # ---------------- SIMULATE ----------------
    t, speed_fpm = build_speed_profile(fs, segments)
    od_mm = simulate_od_from_speed(
        fs=fs,
        t=t,
        speed_fpm=speed_fpm,
        mean_od_mm=mean_od_mm,
        drift_per_ft_mm=drift_per_ft_mm,
        chatter_wavelengths_in=chatter_wavelengths_in,
        chatter_amps_mm=chatter_amps_mm,
        noise_std_running=noise_std_running,
        noise_std_stopped=noise_std_stopped,
    )

    N = len(t)
    # Use current UTC time as start
    start_time = pd.Timestamp(datetime.now())
    t_stamps = make_timestamp_series(start_time, N, fs)

    # Build DataFrames in your desired format
    df_od = pd.DataFrame({
        "t_stamp": t_stamps,
        "Tag_value": od_mm
    })
    df_speed = pd.DataFrame({
        "t_stamp": t_stamps,
        "Tag_value": speed_fpm
    })

    # ---------------- WRITE EXCEL ----------------
    with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
        df_od.to_excel(writer, sheet_name="NDC_System_OD_Value", index=False)
        df_speed.to_excel(writer, sheet_name="YS_Pullout1_Act_Speed_fpm", index=False)

    print(f"Wrote simulated data to {excel_filename}")

if __name__ == "__main__":
    main()
