import numpy as np
import pandas as pd
from datetime import datetime

# ----------------- SPEED PROFILE ----------------- #

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

# ----------------- OD SIMULATION (INCHES) ----------------- #

def simulate_od_from_speed(
    fs,
    t,
    speed_fpm,
    mean_od_in=0.50,           # target OD [inches]
    noise_std_running_in=0.0002,   # noise when line is moving [in]
    noise_std_stopped_in=0.00002,  # noise when line is stopped [in]
    drift_per_ft_in=4.0e-05,       # slow drift per foot of tube [in/ft]
    chatter_wavelengths_in=None,   # list of spatial wavelengths [in]
    chatter_amps_in=None           # list of chatter amplitudes [in]
):
    """
    Simulate OD measurement (in inches) given time, speed profile, and spatial chatter info.
    """
    if chatter_wavelengths_in is None:
        chatter_wavelengths_in = []
    if chatter_amps_in is None:
        chatter_amps_in = [0.0] * len(chatter_wavelengths_in)

    assert len(chatter_wavelengths_in) == len(chatter_amps_in), \
        "chatter_wavelengths_in and chatter_amps_in must be same length"

    N = len(t)
    dt = 1.0 / fs

    # Convert speed to inches per second for spatial → temporal mapping
    speed_in_s = speed_fpm * (12.0 / 60.0)  # 1 fpm = 0.2 in/s

    # Cumulative length along the tube [in] and [ft] for drift term
    length_in = np.cumsum(speed_in_s) * dt
    length_ft = length_in / 12.0

    # Base OD in inches = mean + drift along length
    od_in = mean_od_in + drift_per_ft_in * length_ft

    # Add chatter components using phase accumulation
    for lam_in, amp_in in zip(chatter_wavelengths_in, chatter_amps_in):
        # instantaneous frequency f(t) = v_in_s / λ
        freq_t = np.where(lam_in > 0, speed_in_s / lam_in, 0.0)
        phase = 2 * np.pi * np.cumsum(freq_t) / fs
        od_in += amp_in * np.sin(phase)

    # Add noise: lower when speed=0
    moving_mask = speed_fpm > 0.0
    noise = np.zeros(N)
    noise[moving_mask] = np.random.normal(0.0, noise_std_running_in, moving_mask.sum())
    noise[~moving_mask] = np.random.normal(0.0, noise_std_stopped_in, (~moving_mask).sum())
    od_in += noise

    return od_in

def make_timestamp_series(start_time, N, fs):
    """
    Create a pandas Series of timezone-naive timestamps starting at start_time with spacing 1/fs.
    """
    dt_s = 1.0 / fs
    offsets = pd.to_timedelta(np.arange(N) * dt_s, unit="s")
    return start_time + offsets

# ----------------- MAIN ----------------- #

def main():
    # ------------- CONFIG ------------- #
    fs = 2400.0  # sampling frequency [Hz]

    # speed segments: (duration [s], speed [fpm])
    segments = [
        (10.0, 0.0),       # 10 s stopped
        (60.0, 150.0),     # 60 s at 150 fpm
        (10.0, 0.0),       # 10 s stopped
        (60.0, 200.0),     # 60 s at 200 fpm
    ]

    # Product OD & quality (all inches now)
    mean_od_in = 0.50              # 0.5" tube
    drift_per_ft_in = 4.0e-05      # ≈ 0.001 mm/ft originally

    # Spatial chatter patterns: e.g. 0.5" and 2" repeat along the tube
    chatter_wavelengths_in = [0.5, 2.0]   # inches
    chatter_amps_in = [0.002, 0.001]      # inches (~0.05 mm, 0.025 mm)

    noise_std_running_in = 0.0002         # inches (~0.005 mm)
    noise_std_stopped_in = 0.00002        # inches (~0.0005 mm)

    excel_filename = "tubing_simulated_data.xlsx"

    # ------------- SIMULATE ------------- #
    t, speed_fpm = build_speed_profile(fs, segments)
    od_in = simulate_od_from_speed(
        fs=fs,
        t=t,
        speed_fpm=speed_fpm,
        mean_od_in=mean_od_in,
        drift_per_ft_in=drift_per_ft_in,
        chatter_wavelengths_in=chatter_wavelengths_in,
        chatter_amps_in=chatter_amps_in,
        noise_std_running_in=noise_std_running_in,
        noise_std_stopped_in=noise_std_stopped_in,
    )

    N = len(t)
    # timezone-naive start time (Excel doesn't like tz-aware)
    start_time = pd.Timestamp(datetime.now())
    t_stamps = make_timestamp_series(start_time, N, fs)

    # Build DataFrames in your desired format
    df_od = pd.DataFrame({
        "t_stamp": t_stamps,
        "Tag_value": od_in,         # OD in inches
    })
    df_speed = pd.DataFrame({
        "t_stamp": t_stamps,
        "Tag_value": speed_fpm,     # speed in fpm
    })

    # ------------- WRITE EXCEL ------------- #
    with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
        df_od.to_excel(writer, sheet_name="NDC_System_OD_Value", index=False)
        df_speed.to_excel(writer, sheet_name="YS_Pullout1_Act_Speed_fpm", index=False)

    print(f"Wrote simulated data to {excel_filename}")

if __name__ == "__main__":
    main()
