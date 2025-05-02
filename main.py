import pygame
import random
import time
import json
import os
import pyautogui
screen_size_x, screen_size_y = pyautogui.size()

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((screen_size_x, screen_size_y))
pygame.display.set_caption("Mouse Movement Recorder & Replayer")
clock = pygame.time.Clock()

# Circle settings
circle_radius = 20

# Data storage
all_data = []
data_dir = "mouse_data"
os.makedirs(data_dir, exist_ok=True)

# Generate a random circle position
def random_circle():
    return random.randint(circle_radius, screen_size_x - circle_radius), random.randint(circle_radius, screen_size_y - circle_radius)

# Draw a frame
def draw_frame(target):
    screen.fill((0, 0, 0))
    pygame.draw.circle(screen, (255, 0, 0), target, circle_radius)
    pygame.display.flip()

# Replay function
def replay_from_json(file_path):
    with open(file_path, 'r') as f:
        replay_data = json.load(f)

    for trial in replay_data:
        path = trial['path']
        target = (trial['target_circle']['x'], trial['target_circle']['y'])
        draw_frame(target)

        start_time = path[0][0]
        for i in range(1, len(path)):
            prev_t, prev_x, prev_y = path[i - 1]
            t, x, y = path[i]
            delay = t - prev_t
            pygame.draw.circle(screen, (0, 255, 0), (int(x), int(y)), 3)
            pygame.display.flip()
            time.sleep(delay)

        time.sleep(0.5)

# Game loop
running = True
replay_mode = False
replay_file = None

# Ask user for replay file
if os.path.exists(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    if files:
        print("Available replay files:")
        for i, f in enumerate(files):
            print(f"{i + 1}. {f}")
        choice = input("Enter file number to replay or press Enter to play the game: ")
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            replay_mode = True
            replay_file = os.path.join(data_dir, files[int(choice) - 1])

if replay_mode:
    print(f"Replaying from {replay_file}...")
    replay_from_json(replay_file)
else:
    target = random_circle()
    path = []
    trial_count = 0
    max_trials = 100

    print("Game started. Click on the red circles.")

    while running and trial_count < max_trials:
        draw_frame(target)
        clock.tick(60)

        # Record mouse position and timestamp
        pos = pygame.mouse.get_pos()
        timestamp = time.time()
        path.append((timestamp, pos[0], pos[1]))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                dx = pos[0] - target[0]
                dy = pos[1] - target[1]
                if dx * dx + dy * dy <= circle_radius * circle_radius:
                    # Valid click inside the circle
                    trial_data = {
                        "trial": trial_count,
                        "start_time": path[0][0],
                        "click_time": timestamp,
                        "start_pos": (path[0][1], path[0][2]),
                        "end_pos": (pos[0], pos[1]),
                        "path": path,
                        "target_circle": {
                            "x": target[0],
                            "y": target[1],
                            "radius": circle_radius
                        }
                    }
                    all_data.append(trial_data)
                    print(f"Trial {trial_count} recorded. Circle clicked!")

                    # Reset for next trial
                    path = []
                    target = random_circle()
                    trial_count += 1

    pygame.quit()

    # Save all data to a JSON file
    filename = os.path.join(data_dir, f"mouse_data_{int(time.time())}.json")
    with open(filename, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"Game ended. {trial_count} trials saved to {filename}")
