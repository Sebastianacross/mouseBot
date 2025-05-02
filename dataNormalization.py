import json
import numpy as np
import os
import pyautogui

def normalize_trial(trial):
    path = trial["path"]

    # Convert to numpy for easy manipulation
    path_array = np.array(path)  # shape: (N, 3) = [timestamp, x, y]

    max_dt = 1 / 60  # Normalize using known frame rate (60 FPS) ≈ 0.0167

    # Step 1: convert to deltas
    deltas = np.diff(path_array, axis=0)
    dt = deltas[:, 0:1]
    dx = deltas[:, 1:2]
    dy = deltas[:, 2:3]

    # Step 2: normalize dx, dy by screen size
    screen_size_x, screen_size_y = pyautogui.size()
    dx /= screen_size_x  # assuming width = 800
    dy /= screen_size_y  # assuming height = 600

    # Step 3: normalize dt using max_dt
    dt = dt / max_dt

    # Combine into (N-1, 3) sequence
    normalized = np.hstack([dt, dx, dy])

    return normalized.tolist()

def normalize_all_trials(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    normalized_data = []
    for trial in data:
        norm_path = normalize_trial(trial)
        normalized_data.append({
            "trial": trial["trial"],
            "start_pos": trial["start_pos"],
            "end_pos": trial["end_pos"],
            "target_circle": trial["target_circle"],
            "normalized_path": norm_path
        })

    with open(output_file, 'w') as f:
        json.dump(normalized_data, f, indent=2)

    print(f"✅ Normalized data saved to {output_file}")

# Example usage
data_dir = "mouse_data_normalized"
os.makedirs(data_dir, exist_ok=True)
input_json = "mouse_data/mouse_data_1743325848.json"  # use your filename
output_json = "mouse_data_normalized/normalized_mouse_data_1743325848.json"
normalize_all_trials(input_json, output_json)
