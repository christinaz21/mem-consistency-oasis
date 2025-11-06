from collections import deque
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

class ExploreAgent:
    def __init__(self, min_length=5, debug=False):
        self.min_length = min_length
        self.debug = debug
    
    def reset(self, maze_layout, start_pos):
        self.maze_layout = maze_layout
        discrete_start_pos = (int(start_pos[1]), int(start_pos[0]))
        while True:
            target_pos = (np.random.randint(maze_layout.shape[0]),
                          np.random.randint(maze_layout.shape[1]))
            path = self.BFS(discrete_start_pos, target_pos)
            if path is not None:
                if len(path) < self.min_length:
                    continue
                break

        new_path = [start_pos]
        for pos in path[1:]:
            new_path.append((pos[1] + 0.5, pos[0] + 0.5))
        self.path = new_path

    def BFS(self, start, target):
        if self.debug:
            print("BFS from", start, "to", target)
        if self.maze_layout[target] == 0:
            return None  # Target is a wall
        rows, cols = self.maze_layout.shape
        visited = set()
        queue = deque([(start, [start])])

        while queue:
            (current, path) = queue.popleft()
            if current == target:
                return path

            for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and
                        self.maze_layout[neighbor] == 1 and neighbor not in visited):
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None
    
    def step(self, current_pos, current_dir):
        # actions: (noop, forward, left, right, forward_left, forward_right)
        # Advance to the next waypoint if close enough
        while len(self.path) > 1 and np.linalg.norm(np.array(self.path[0]) - np.array(current_pos)) < 0.1:
            self.path.pop(0)

        # Compute desired direction
        target = self.path[0]
        delta = np.array([target[0] - current_pos[0], target[1] - current_pos[1]])
        norm = np.linalg.norm(delta)
        if norm > 0:
            desired_dir = delta / norm
        else:
            desired_dir = np.zeros(2)

        # Compute steering angle between current_dir and desired_dir
        dot = np.dot(current_dir, desired_dir)
        cross = current_dir[0] * desired_dir[1] - current_dir[1] * desired_dir[0]
        angle = np.arctan2(cross, dot)

        # Choose action based on angle
        if abs(angle) < np.deg2rad(10):
            action = 1  # forward
        elif angle > 0:
            if abs(angle) < np.deg2rad(30):
                action = 4  # forward_left
            else:
                action = 2  # left
        else:
            if abs(angle) < np.deg2rad(30):
                action = 5  # forward_right
            else:
                action = 3  # right
        if abs(current_pos[0] - self.path[-1][0]) + abs(current_pos[1] - self.path[-1][1]) < 0.2:
            return True, action
        else:
            return False, action

def collect(file_idx, root_dir, env, agent, save_fig=False, debug=False):
    obs = env.reset()
    if debug:
        print("Agent start position:", obs["agent_pos"])

    agent.reset(obs["maze_layout"], obs["agent_pos"])
    if save_fig:
        img_save_path = f"/home/tc0786/Project/train-oasis/data/maze_9/visualization/self_eval_{file_idx}.png"
        os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
        plt.imshow(obs["maze_layout"], cmap='gray')
        for idx in range(len(agent.path) - 1):
            point_0 = agent.path[idx]
            point_1 = agent.path[idx + 1]
            plt.plot([point_0[0] - 0.5, point_1[0] - 0.5], [point_0[1] - 0.5, point_1[1] - 0.5], 'r-')
        plt.plot(agent.path[0][0] - 0.5, agent.path[0][1] - 0.5, 'go', markersize=8, label='start')
        plt.plot(agent.path[-1][0] - 0.5, agent.path[-1][1] - 0.5, 'ro', markersize=8, label='target')
    
    global_step = 0
    backward_path = agent.path.copy()
    backward_path.reverse()
    all_data = []
    first_action = np.zeros(6, dtype=np.float64)
    obs["action"] = first_action
    all_data.append(obs)
    # forward
    while True:
        finished, action = agent.step(obs["agent_pos"], obs["agent_dir"])
        obs, reward, done, info = env.step(action)
        one_hot_action = np.zeros(6, dtype=np.float64)
        one_hot_action[action] = 1.0
        obs["action"] = one_hot_action
        all_data.append(obs)
        if finished:
            break
        global_step += 1
        if global_step > 500:
            print("Forward path too long, aborting.")
            return None
    # backward
    agent.path = backward_path
    global_step = 0
    while True:
        finished, action = agent.step(obs["agent_pos"], obs["agent_dir"])
        obs, reward, done, info = env.step(action)
        one_hot_action = np.zeros(6, dtype=np.float64)
        one_hot_action[action] = 1.0
        obs["action"] = one_hot_action
        all_data.append(obs)
        if finished:
            break
        global_step += 1
        if global_step > 500:
            print("Backward path too long, aborting.")
            return None
    
    # save data
    data_save_path = os.path.join(root_dir, f"{file_idx:04d}.npz")
    np.savez_compressed(data_save_path, **{k: np.array([d[k] for d in all_data]) for k in all_data[0].keys()})

    if save_fig:
        for idx in range(len(all_data) - 1):
            point_0 = all_data[idx]["agent_pos"]
            point_1 = all_data[idx + 1]["agent_pos"]
            plt.plot([point_0[0] - 0.5, point_1[0] - 0.5], [point_0[1] - 0.5, point_1[1] - 0.5], 'b-')
        plt.title("Explore Agent Path")
        plt.savefig(img_save_path)
        plt.close()
    
    return len(all_data)

def collect_batch(file_idx, root_dir, env, agent, save_fig=False, debug=False):
    obs = env.reset()
    if debug:
        print("Agent start position:", obs["agent_pos"])

    length_range = (13, 15)
    try_count = 20
    now_count = 0
    while True:
        if now_count >= try_count:
            print("Failed to find a suitable path length, aborting.")
            return None
        agent.reset(obs["maze_layout"], obs["agent_pos"])
        path_length = len(agent.path)
        if length_range[0] <= path_length <= length_range[1]:
            break
        now_count += 1
    if save_fig:
        img_save_path = f"/home/tc0786/Project/train-oasis/data/maze/visualization/self_eval_{file_idx}.png"
        os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
        plt.imshow(obs["maze_layout"], cmap='gray')
        for idx in range(len(agent.path) - 1):
            point_0 = agent.path[idx]
            point_1 = agent.path[idx + 1]
            plt.plot([point_0[0] - 0.5, point_1[0] - 0.5], [point_0[1] - 0.5, point_1[1] - 0.5], 'r-')
        plt.plot(agent.path[0][0] - 0.5, agent.path[0][1] - 0.5, 'go', markersize=8, label='start')
        plt.plot(agent.path[-1][0] - 0.5, agent.path[-1][1] - 0.5, 'ro', markersize=8, label='target')
    
    global_step = 0
    backward_path = agent.path.copy()
    backward_path.reverse()
    all_data = []
    first_action = np.zeros(6, dtype=np.float64)
    obs["action"] = first_action
    all_data.append(obs)
    # forward
    while True:
        finished, action = agent.step(obs["agent_pos"], obs["agent_dir"])
        obs, reward, done, info = env.step(action)
        one_hot_action = np.zeros(6, dtype=np.float64)
        one_hot_action[action] = 1.0
        obs["action"] = one_hot_action
        all_data.append(obs)
        if finished:
            break
        global_step += 1
        if global_step > 500:
            print("Forward path too long, aborting.")
            return None
    # backward
    agent.path = backward_path
    global_step = 0
    while True:
        finished, action = agent.step(obs["agent_pos"], obs["agent_dir"])
        obs, reward, done, info = env.step(action)
        one_hot_action = np.zeros(6, dtype=np.float64)
        one_hot_action[action] = 1.0
        obs["action"] = one_hot_action
        all_data.append(obs)
        if finished:
            break
        global_step += 1
        if global_step > 500:
            print("Backward path too long, aborting.")
            return None
    
    # save data
    data_save_path = os.path.join(root_dir, f"{file_idx:04d}.npz")
    np.savez_compressed(data_save_path, **{k: np.array([d[k] for d in all_data]) for k in all_data[0].keys()})

    if save_fig:
        for idx in range(len(all_data) - 1):
            point_0 = all_data[idx]["agent_pos"]
            point_1 = all_data[idx + 1]["agent_pos"]
            plt.plot([point_0[0] - 0.5, point_1[0] - 0.5], [point_0[1] - 0.5, point_1[1] - 0.5], 'b-')
        plt.title("Explore Agent Path")
        plt.savefig(img_save_path)
        plt.close()
    
    return len(all_data)

def create_evaluation_maze():
    env = gym.make('memory_maze:MemoryMaze-15x15-ExtraObs-v0')
    agent = ExploreAgent()
    metadata = []
    num_mazes = 200
    save_fig_num = 20
    root_dir = "/home/tc0786/Project/train-oasis/data/maze_9/self_eval"
    os.makedirs(root_dir, exist_ok=True)
    for i in tqdm(range(num_mazes)):
        l = collect(i, root_dir, env, agent, save_fig=(i < save_fig_num))
        if l is None:
            continue
        metadata.append({
            "file": os.path.join(root_dir, f"{i:04d}.npz"),
            "length": l,
        })
    with open("/home/tc0786/Project/train-oasis/data/maze_9/metadata_self.json", "w") as f:
        json.dump(metadata, f, indent=4)

def create_evaluation_maze_batch():
    env = gym.make('memory_maze:MemoryMaze-15x15-ExtraObs-v0')
    agent = ExploreAgent()
    metadata = []
    num_mazes = 200
    save_fig_num = 20
    root_dir = "/home/tc0786/Project/train-oasis/data/maze/self_eval_batch"
    os.makedirs(root_dir, exist_ok=True)
    for i in tqdm(range(num_mazes)):
        while True:
            l = collect_batch(i, root_dir, env, agent, save_fig=(i < save_fig_num))
            if l is not None:
                break
            print("Failed to collect data, retrying...")
        metadata.append({
            "file": os.path.join(root_dir, f"{i:04d}.npz"),
            "length": l,
        })
    with open("/home/tc0786/Project/train-oasis/data/maze/metadata_self_batch.json", "w") as f:
        json.dump(metadata, f, indent=4)

def check_data():
    path = "/home/tc0786/Project/train-oasis/data/maze_9/self_eval/0000.npz"
    data = np.load(path, allow_pickle=True)
    print(data.files)
    for file in data.files:
        print(file)
        print(data[file].shape, data[file].dtype)
        print(data[file][0])
        print()

def check_length():
    env = gym.make('memory_maze:MemoryMaze-15x15-ExtraObs-v0')
    agent = ExploreAgent()
    total_num = 200
    all_lengths = []
    for i in range(total_num):
        obs = env.reset()
        agent.reset(obs["maze_layout"], obs["agent_pos"])
        length = len(agent.path)
        all_lengths.append(length)
        print(f"Maze {i}: path length = {length}")

    print(f"Average path length over {total_num} mazes: {np.mean(all_lengths)}")

def handle_batch():
    metadata_path = "/home/tc0786/Project/train-oasis/data/maze/metadata_self_batch.json"
    output_path = "/home/tc0786/Project/train-oasis/data/maze/metadata_self_batch_processed.json"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    lengths = [item["length"] for item in metadata]
    lengths = sorted(lengths)
    for total_lengths in range(20, 40):
        min_gap = float('inf')
        flag = -1
        for start_idx in range(len(lengths) - total_lengths + 1):
            if lengths[start_idx + total_lengths - 1] - lengths[start_idx] < min_gap:
                min_gap = lengths[start_idx + total_lengths - 1] - lengths[start_idx]
                flag = start_idx

        print(f"Total lengths: {total_lengths}, Min gap: {min_gap}, Start length: {lengths[flag]}, End length: {lengths[flag + total_lengths - 1]}")


if __name__ == "__main__":
    # create_evaluation_maze()
    # check_data()
    # check_length()
    create_evaluation_maze_batch()
    # handle_batch()