import pyautogui
import time
import torch
import numpy as np
import random
from mouse_transformer_model import MouseTransformer
from scipy.interpolate import CubicSpline
# Define model class (must match training script)
class MouseLSTM(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2, dropout=0.2):
        super(MouseLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers,
                                  batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, 3)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


def move_mouse_like_human_to_target(checkpoint_path, target_pos,
                                    min_steps=30, max_steps=120,
                                    stop_threshold=10,
                                    seed_sequence=None,
                                    device='cpu'):
    # Load model and avg_dt from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    avg_dt = checkpoint['avg_dt']

    model = MouseLSTM()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    start_pos = pyautogui.position()
    current_pos = np.array((start_pos.x, start_pos.y), dtype=np.float32)
    current_time = 0.0

    seed_len = 20
    if seed_sequence is None:
        seed = torch.zeros((1, seed_len, 3), dtype=torch.float32).to(device)
    else:
        seed = torch.tensor(seed_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    steps = random.randint(min_steps, max_steps)
    path = []

    # --- Tuning Parameters ---
    MOVEMENT_SCALE = 3       # Slightly lower for less jumpy motion
    STEERING_WEIGHT = 0.95     # How much to trust the target direction vs the model
    STEERING_FORCE = 50        # How strongly to push toward the target in pixels per step
    SMOOTHING_FACTOR = 0.85    # More inertia = smoother curves
    JITTER = 0.2               # Subtle natural tremor
    SUBSTEPS = 2               # Number of interpolation steps per move

    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

    prev_velocity = np.array([0.0, 0.0])  # for velocity smoothing
    overshoot_done = False

    for _ in range(steps):
        with torch.no_grad():
            delta = model(seed).squeeze().cpu().numpy()

        dt, dx, dy = delta
        current_time += dt

        model_dx = dx * SCREEN_WIDTH * MOVEMENT_SCALE
        model_dy = dy * SCREEN_HEIGHT * MOVEMENT_SCALE

        to_target = np.array(target_pos, dtype=np.float32) - current_pos
        distance_to_target = np.linalg.norm(to_target)
        to_target_norm = to_target / (distance_to_target + 1e-8)

        blended_dx = (1 - STEERING_WEIGHT) * model_dx + STEERING_WEIGHT * to_target_norm[0] * STEERING_FORCE
        blended_dy = (1 - STEERING_WEIGHT) * model_dy + STEERING_WEIGHT * to_target_norm[1] * STEERING_FORCE

        scaling = np.clip(distance_to_target / 300, 0.2, 1.0)
        blended_dx *= scaling
        blended_dy *= scaling

        velocity = np.array([blended_dx, blended_dy])

        # --- Cap abrupt angle changes ---
        angle = np.arccos(
            np.clip(np.dot(prev_velocity, velocity) / (
                np.linalg.norm(prev_velocity) * np.linalg.norm(velocity) + 1e-8), -1.0, 1.0))
        if angle > np.radians(45):
            velocity = 0.9 * prev_velocity + 0.1 * velocity

        # Apply inertia smoothing
        smoothed_velocity = SMOOTHING_FACTOR * prev_velocity + (1 - SMOOTHING_FACTOR) * velocity
        prev_velocity = smoothed_velocity.copy()

        jitter = np.random.uniform(-JITTER, JITTER, size=2)
        movement = smoothed_velocity + jitter
        step_size = np.linalg.norm(movement)

        if step_size < 1.5:
            continue

        next_pos = current_pos + movement

        # --- Subpixel smoothing with interpolation ---
        for i in range(1, SUBSTEPS + 1):
            interp_pos = current_pos + (next_pos - current_pos) * (i / SUBSTEPS)
            pyautogui.moveTo(interp_pos[0], interp_pos[1])
            TIME_SCALE = 1
            time.sleep(float(max((dt * 0.0167 / SUBSTEPS) * TIME_SCALE, 0.001)))

        current_pos = next_pos
        path.append((current_time, current_pos[0], current_pos[1]))

        print(f"step: {step_size:.2f}px, to_target: {distance_to_target:.1f}px, dx: {dx:.4f}, dy: {dy:.4f}")

        if distance_to_target < stop_threshold and not overshoot_done:
            # --- Add slight overshoot ---
            overshoot_vector = to_target_norm * random.uniform(5, 15)
            overshoot_pos = current_pos + overshoot_vector
            pyautogui.moveTo(overshoot_pos[0], overshoot_pos[1], duration=0.05)
            pyautogui.moveTo(target_pos[0], target_pos[1], duration=0.05)
            overshoot_done = True
            break

        new_step = torch.tensor([[dt, dx, dy]], dtype=torch.float32).to(device)
        seed = torch.cat([seed[:, 1:, :], new_step.unsqueeze(0)], dim=1)

    # --- Add small pause before clicking ---
    time.sleep(random.uniform(0.1, 0.2))
    pyautogui.click()
    print(f"✅ Mouse moved to {target_pos} and clicked.")
    return path


def move_mouse_like_gamer_Transformer(checkpoint_path, target_pos,
                          min_steps=6, max_steps=100,
                          stop_threshold=8,
                          seed_sequence=None,
                          device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    avg_dt = checkpoint['avg_dt']

    model = MouseTransformer()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    start_pos = pyautogui.position()
    current_pos = np.array((start_pos.x, start_pos.y), dtype=np.float32)
    current_time = 0.0

    seed_len = 20
    if seed_sequence is None:
        seed = torch.zeros((1, seed_len, 3), dtype=torch.float32).to(device)
    else:
        seed = torch.tensor(seed_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    path = []
    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
    MOVEMENT_SCALE = 10.0
    STEERING_FORCE = 90
    GAMER_SMOOTHING = 0.35

    prev_velocity = np.array([0.0, 0.0])
    current_time = 0.0

    start_time = time.time()

    for step_num in range(max_steps):
        with torch.no_grad():
            delta = model(seed).squeeze().cpu().numpy()

        dt, dx, dy = delta
        current_time += dt

        model_dx = dx * SCREEN_WIDTH * MOVEMENT_SCALE
        model_dy = dy * SCREEN_HEIGHT * MOVEMENT_SCALE

        to_target = np.array(target_pos, dtype=np.float32) - current_pos
        distance_to_target = np.linalg.norm(to_target)
        to_target_norm = to_target / (distance_to_target + 1e-8)

        steer_weight = np.clip(distance_to_target / 120, 0.1, 0.95)
        blended_dx = (1 - steer_weight) * model_dx + steer_weight * to_target_norm[0] * STEERING_FORCE
        blended_dy = (1 - steer_weight) * model_dy + steer_weight * to_target_norm[1] * STEERING_FORCE

        velocity = np.array([blended_dx, blended_dy])
        smoothed_velocity = GAMER_SMOOTHING * prev_velocity + (1 - GAMER_SMOOTHING) * velocity
        prev_velocity = smoothed_velocity.copy()

        progress = step_num / max_steps
        speed_scale = np.sin(np.pi * progress)
        smoothed_velocity *= speed_scale

        next_pos = current_pos + smoothed_velocity
        current_pos = next_pos

        path.append((current_time, current_pos[0], current_pos[1]))

        if distance_to_target < stop_threshold:
            break

        new_step = torch.tensor([[dt, dx, dy]], dtype=torch.float32).to(device)
        seed = torch.cat([seed[:, 1:, :], new_step.unsqueeze(0)], dim=1)

    # --- Add target to path for final smoothing ---
    path.append((current_time + 0.01, target_pos[0], target_pos[1]))
    path_np = np.array(path)

    # --- Spline smoothing with fast exponential easing ---
    if len(path_np) >= 4:
        t_vals = np.linspace(0, 1, len(path_np))
        cs_x = CubicSpline(t_vals, path_np[:, 1])
        cs_y = CubicSpline(t_vals, path_np[:, 2])

        #interp_points = min(len(path_np) * 2, 20)

        interp_points = 12
        t_interp = np.geomspace(0.001, 1, interp_points)
        t_interp = (t_interp - t_interp.min()) / (t_interp.max() - t_interp.min())

        for x, y in zip(cs_x(t_interp), cs_y(t_interp)):
            pyautogui.moveTo(x, y)
            # minimal delay for sub-second execution
            #time.sleep(0.00002)
    else:
        for _, x, y in path_np:
            pyautogui.moveTo(x, y)
            #time.sleep(0.00002)

    time.sleep(0.01)
    pyautogui.click()

    print(f"✅ Ultra-fluid mouse moved to {target_pos} in {time.time() - start_time:.3f}s.")
    return path



# move_mouse_like_human_to_target(
#     checkpoint_path="models/mouse_lstm_checkpoint.pth",
#     target_pos=(692, 813),
#     device='cuda' if torch.cuda.is_available() else 'cpu'
# )

move_mouse_like_gamer_Transformer(
    checkpoint_path="models/mouse_transformer.pth",
    target_pos=(692, 813),
    device='cuda' if torch.cuda.is_available() else 'cpu'
)