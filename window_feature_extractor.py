import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.metrics import roc_curve, auc

FRAME_RATE = 30
AGE_MAX_WKS = 1000
AGE_MIN_WKS = 0


def window_feature_extractor(mice_info_dir, input_dir, output_dir,
                             window_length_min=1.0,
                             step_min=0.5,
                             max_normal=None,
                             max_blind=None):
    """
    Calculate multiple behavior metrics for each mouse within each sliding window

    Input file format requirements:
    - Line 1: Light:X=..., Y=..., W=..., H=...; Dark:X=..., Y=..., W=..., H=...
    - Subsequent lines: frame_number, x, y

    Output file: each line represents a sliding window with multiple metrics.
    """

    os.makedirs(output_dir, exist_ok=True)
    window_size = int(window_length_min * 60 * FRAME_RATE)
    step_size = int(step_min * 60 * FRAME_RATE)

    # Actual light chamber dimensions (in cm)
    LIGHT_REAL_WIDTH_CM = 40.64
    LIGHT_REAL_HEIGHT_CM = 30.48

    MAX_VALID_STEP_CM = 2.0  # Maximum movement threshold between two frames, exceeding this is considered abnormal
    IMMOBILE_FRAME_COUNT = 30  # Consecutive frame count, equals 1 second
    IMMOBILE_RADIUS_CM = 1.0  # Maximum movement radius (immobility threshold)

    df_info = pd.read_excel(mice_info_dir)
    df_info["Number"] = df_info["Number"].astype(int)
    info_lookup = {int(row["Number"]): row for _, row in df_info.iterrows()}

    np.random.seed(0)

    # First collect all eligible mouse files, group by type
    normal_files = []
    blind_files = []

    for fname in os.listdir(input_dir):
        if not fname.endswith(".txt"):
            continue

        try:
            mouse_number = int(fname.split("_")[0])
        except:
            print(f"Skip file {fname}: number cannot be parsed")
            continue

        if mouse_number not in info_lookup:
            print(f"Skip file {fname}: number {mouse_number} not found in Excel")
            continue

        info = info_lookup[mouse_number]
        type_str = str(info["Type"]).strip()
        blind_str = str(info["Blind/Normal"]).strip()
        age_str = str(info["Age"]).strip().replace(" ", "")
        gender_str = str(info["Sex"]).strip()

        match_age = re.match(r"(\d+)", age_str)
        if not match_age:
            print(f"[Skip] {fname}: cannot parse age {age_str}")
            continue
        age_weeks = int(match_age.group(1))

        if age_weeks < AGE_MIN_WKS or age_weeks > AGE_MAX_WKS:
            print(f"[Skip processing] {fname}: age {age_weeks} weeks")
            continue

        # Group and collect files by Blind/Normal type
        if blind_str.lower() == "normal":
            normal_files.append((fname, mouse_number, info))
        elif blind_str.lower() == "blind":
            blind_files.append((fname, mouse_number, info))

    # Randomly select specified number of mice
    if max_normal is not None and len(normal_files) > max_normal:
        selected_normal = np.random.choice(len(normal_files), size=max_normal, replace=False)
        selected_normal_files = [normal_files[i] for i in selected_normal]
        print(f"Randomly selected {max_normal} from {len(normal_files)} normal mice")
    else:
        selected_normal_files = normal_files
        print(f"Selected all {len(normal_files)} normal mice")

    if max_blind is not None and len(blind_files) > max_blind:
        selected_blind = np.random.choice(len(blind_files), size=max_blind, replace=False)
        selected_blind_files = [blind_files[i] for i in selected_blind]
        print(f"Randomly selected {max_blind} from {len(blind_files)} blind mice")
    else:
        selected_blind_files = blind_files
        print(f"Selected all {len(blind_files)} blind mice")

    selected_files = selected_normal_files + selected_blind_files

    # Count and print selected mouse types
    wt_ages = []
    double_crushed_rho_ko_ages = []
    young_rho_ko_ages = []
    old_rho_ko_ages = []

    for fname, mouse_number, info in selected_files:
        type_str = str(info["Type"]).strip()
        blind_str = str(info["Blind/Normal"]).strip()
        note_str = str(info.get("Note", "")).strip() if "Note" in info else ""
        age_str = str(info["Age"]).strip().replace(" ", "")

        # Extract age
        match_age = re.match(r"(\d+)", age_str)
        age_weeks = int(match_age.group(1))

        if type_str.lower() == "wt":
            wt_ages.append(age_weeks)
        elif type_str.lower() == "rho ko":
            if "crash" in note_str.lower():
                double_crushed_rho_ko_ages.append(age_weeks)
            elif blind_str.lower() == "normal":  # young RKO (normal vision)
                young_rho_ko_ages.append(age_weeks)
            elif blind_str.lower() == "blind":  # old RKO (blind but not crushed)
                old_rho_ko_ages.append(age_weeks)

    print(f"\n=== Selected mouse type counts and age ranges ===")

    if wt_ages:
        wt_min, wt_max = min(wt_ages), max(wt_ages)
        print(f"WT: {len(wt_ages)} mice, age range: {wt_min}-{wt_max} weeks")
    else:
        print(f"WT: 0 mice")

    if double_crushed_rho_ko_ages:
        dc_min, dc_max = min(double_crushed_rho_ko_ages), max(double_crushed_rho_ko_ages)
        print(f"Double crushed RKO: {len(double_crushed_rho_ko_ages)} mice, age range: {dc_min}-{dc_max} weeks")
    else:
        print(f"Double crushed RKO: 0 mice")

    if young_rho_ko_ages:
        young_min, young_max = min(young_rho_ko_ages), max(young_rho_ko_ages)
        print(f"Young RKO (normal vision): {len(young_rho_ko_ages)} mice, age range: {young_min}-{young_max} weeks")
    else:
        print(f"Young RKO (normal vision): 0 mice")

    if old_rho_ko_ages:
        old_min, old_max = min(old_rho_ko_ages), max(old_rho_ko_ages)
        print(f"Old RKO (blind, not crushed): {len(old_rho_ko_ages)} mice, age range: {old_min}-{old_max} weeks")
    else:
        print(f"Old RKO (blind, not crushed): 0 mice")

    print(f"Total: {len(selected_files)} mice")
    print("=" * 50)

    # Process selected files
    for fname, mouse_number, info in selected_files:
        type_str = str(info["Type"]).strip()
        blind_str = str(info["Blind/Normal"]).strip()
        age_str = str(info["Age"]).strip().replace(" ", "")
        gender_str = str(info["Sex"]).strip()

        match_age = re.match(r"(\d+)", age_str)
        age_weeks = int(match_age.group(1))
        age_weeks += np.random.uniform(-0.5, 0.5)

        input_path = os.path.join(input_dir, fname)
        with open(input_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Parse light and dark chambers
        match = re.search(
            r"Light:X=(\d+), Y=(\d+), W=(\d+), H=(\d+);\s*Dark:X=(\d+), Y=(\d+), W=(\d+), H=(\d+)",
            lines[0])
        if not match:
            print(f"[Skip] {fname}: first line missing light/dark chamber information")
            continue
        lx, ly, lw, lh = map(int, match.groups()[0:4])
        dx, dy, dw, dh = map(int, match.groups()[4:8])

        # Pixel to cm ratio (calculated using light chamber only)
        cm_per_pixel_x = LIGHT_REAL_WIDTH_CM / lw
        cm_per_pixel_y = LIGHT_REAL_HEIGHT_CM / lh

        coords = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    x = int(parts[1].strip())
                    y = int(parts[2].strip())
                    coords.append((x, y))
                except:
                    continue

        if len(coords) < window_size:
            print(f"Skip file {fname}: insufficient frames")
            continue

        def get_side(x, y):
            if lx <= x <= lx + lw and ly <= y <= ly + lh:
                return "light"
            elif dx <= x <= dx + dw and dy <= y <= dy + dh:
                return "dark"
            else:
                return "unknown"

        results = []

        for start in range(0, len(coords) - window_size + 1, step_size):
            sub_coords = coords[start:start + window_size]
            valid = [(x, y, get_side(x, y)) for (x, y) in sub_coords if (x, y) != (-1, -1)]

            if not valid:
                print(f"[Skip] {fname} window at {start / FRAME_RATE:.1f}s: no valid trajectory")
                continue

            side_seq = [s for (_, _, s) in valid if s in ("light", "dark")]
            if len(side_seq) == 0:
                continue

            # Metric 1: dark chamber percentage
            num_dark = side_seq.count("dark")
            dark_pct = (num_dark / len(side_seq)) * 100

            # Metric 2: movement distance in light chamber consecutive frames (cm)
            light_distance_cm = 0.0
            light_valid_frame_count = 0  # for calculating light chamber average speed
            for i in range(1, len(valid)):
                (x1, y1, side1) = valid[i - 1]
                (x2, y2, side2) = valid[i]
                if side1 == "light" and side2 == "light":
                    dx_cm = (x2 - x1) * cm_per_pixel_x
                    dy_cm = (y2 - y1) * cm_per_pixel_y
                    dist = (dx_cm ** 2 + dy_cm ** 2) ** 0.5
                    if dist < MAX_VALID_STEP_CM:
                        light_distance_cm += dist
                        light_valid_frame_count += 1

            # Metric 3: movement distance in dark chamber consecutive frames (cm)
            dark_distance_cm = 0.0
            dark_valid_frame_count = 0  # for calculating dark chamber average speed
            for i in range(1, len(valid)):
                (x1, y1, side1) = valid[i - 1]
                (x2, y2, side2) = valid[i]
                if side1 == "dark" and side2 == "dark":
                    dx_cm = (x2 - x1) * cm_per_pixel_x
                    dy_cm = (y2 - y1) * cm_per_pixel_y
                    dist = (dx_cm ** 2 + dy_cm ** 2) ** 0.5
                    if dist < MAX_VALID_STEP_CM:
                        dark_distance_cm += dist
                        dark_valid_frame_count += 1

            # Metric 4: number of light → dark crossings
            cross_count = 0
            for i in range(1, len(side_seq)):
                if side_seq[i - 1] == "light" and side_seq[i] == "dark":
                    cross_count += 1

            # Metrics 5+6: immobile time in light and dark chambers (seconds)
            immobile_time_light_s = 0
            immobile_time_dark_s = 0
            i = 0
            while i <= len(valid) - IMMOBILE_FRAME_COUNT:
                (ref_x, ref_y, ref_side) = valid[i]
                if ref_side not in ("light", "dark"):
                    i += 1
                    continue

                all_within = True
                for j in range(1, IMMOBILE_FRAME_COUNT):
                    xj, yj, sidej = valid[i + j]
                    if sidej != ref_side:
                        all_within = False
                        break

                    dx_cm = (xj - ref_x) * cm_per_pixel_x
                    dy_cm = (yj - ref_y) * cm_per_pixel_y
                    dist = (dx_cm ** 2 + dy_cm ** 2) ** 0.5
                    if dist > IMMOBILE_RADIUS_CM:
                        all_within = False
                        break

                if all_within:
                    if ref_side == "light":
                        immobile_time_light_s += 1
                    elif ref_side == "dark":
                        immobile_time_dark_s += 1
                    i += IMMOBILE_FRAME_COUNT  # skip this segment
                else:
                    i += 1

            # Metric 7: light chamber average speed
            light_avg_speed = 0.0
            if light_valid_frame_count > 0:
                light_avg_speed = light_distance_cm / (light_valid_frame_count / FRAME_RATE)
            else:
                light_avg_speed = 0.0

            # Metric 8: dark chamber average speed
            dark_avg_speed = 0.0
            if dark_valid_frame_count > 0:
                dark_avg_speed = dark_distance_cm / (dark_valid_frame_count / FRAME_RATE)
            else:
                dark_avg_speed = 0.0

            # Metrics 9-10: Standard deviation of latency to dark/light (in seconds)
            # Skip the first segment in each window to avoid truncated delays
            latencies_to_dark = []
            latencies_to_light = []
            # Extract side sequence
            side_seq = [s for (_, _, s) in valid if s in ("light", "dark")]

            # Skip the first consecutive segment (incomplete)
            if not side_seq:
                latency_to_dark_sd = 0.0
                latency_to_light_sd = 0.0
            else:
                current_side = side_seq[0]
                i = 1
                while i < len(side_seq) and side_seq[i] == current_side:
                    i += 1

                # Start counting latency after switches
                duration = 0
                while i < len(side_seq):
                    s = side_seq[i]
                    if s == current_side:
                        duration += 1
                    else:
                        # State switch, record latency of previous segment
                        if current_side == "light" and duration > 0:
                            latencies_to_dark.append(duration / FRAME_RATE)
                        elif current_side == "dark" and duration > 0:
                            latencies_to_light.append(duration / FRAME_RATE)

                        # Reset
                        current_side = s
                        duration = 1
                    i += 1

                # Handle the last segment
                if duration > 0:
                    if current_side == "light":
                        latencies_to_dark.append(duration / FRAME_RATE)
                    elif current_side == "dark":
                        latencies_to_light.append(duration / FRAME_RATE)

                # Calculate standard deviation
                latency_to_dark_sd = np.std(latencies_to_dark) if latencies_to_dark else 0.0
                latency_to_light_sd = np.std(latencies_to_light) if latencies_to_light else 0.0

            results.append([dark_pct, light_distance_cm, dark_distance_cm, cross_count,
                            immobile_time_light_s, immobile_time_dark_s, light_avg_speed, dark_avg_speed,
                            latency_to_dark_sd, latency_to_light_sd])

        save_name = f"{mouse_number}_{type_str}_{blind_str}_{age_str}_{gender_str}_W{int(window_length_min)}min_S{step_min:.2f}min_metrics.txt"
        save_path = os.path.join(output_dir, save_name)
        np.savetxt(save_path, results, fmt="%.2f")
        print(f"Saved: {save_path}")