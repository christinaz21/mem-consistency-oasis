import matplotlib.pyplot as plt
import numpy as np
import json
import os
import cv2
import seaborn as sns
import wandb

def draw_one():
    sns.set_theme(style="whitegrid")  # Set Seaborn theme

    paths = [
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_epoch1.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_epoch2.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_epoch3.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_epoch4.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_epoch5.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_epoch6.json",
    ]
    metric_names = ["mse"]
    save_dir = "outputs/rnn/figs/conflict"
    # save_dir = "/scratch/gpfs/CHIJ/taiye/outputs/draw_figs/compounding_error"

    for metric_name in metric_names:
        plt.figure(figsize=(5, 4))
        all_long = []
        all_short = []
        for path in paths:
            with open(path, "r") as f:
                data = json.load(f)
            metrics = {}
            for item in data:
                prompt_frames = item["prompt_frames"]
                for key, values in item["metrics"].items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].extend(values[prompt_frames:])
            avg_metrics = {key: sum(values) / len(values) for key, values in metrics.items()}
            all_long.append(avg_metrics[metric_name])
            path = path.replace("eval_outputs", "abla_eval_outputs")
            with open(path, "r") as f:
                data = json.load(f)
            metrics = {}
            for item in data:
                prompt_frames = item["prompt_frames"]
                for key, values in item["metrics"].items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].extend(values[prompt_frames:])
            avg_metrics = {key: sum(values) / len(values) for key, values in metrics.items()}
            all_short.append(avg_metrics[metric_name])
        # Apply smoothing with configurable window size
        sns.lineplot(x=np.arange(1, 1+len(all_long)), y=all_long, label="Long-Term", linewidth=3.5)
        sns.lineplot(x=np.arange(1, 1+len(all_short)), y=all_short, label="Short-Term", linewidth=3.5)
        y_max = max(max(all_long), max(all_short))
        y_min = min(min(all_long), min(all_short))
        y_lim_high = y_max + 0.3 * (y_max - y_min)
        plt.ylim(top=y_lim_high)

        plt.xlabel("Epoch", fontsize=24)
        plt.ylabel(metric_name.upper(), fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=14, loc="upper right", ncol=2)
        plt.grid(True)
        
        # Add arrow to x-axis
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        
        # Add larger arrow to x-axis
        ax.annotate('', xy=(1.02, 0), xytext=(-0.02, 0),
                    xycoords=('axes fraction', 'axes fraction'),
                    textcoords=('axes fraction', 'axes fraction'),
                    arrowprops=dict(arrowstyle='->', color='black', lw=3, 
                                    mutation_scale=20))

        # Set x-axis to start from 1 and offset ticks by 1
        plt.xlim(left=1)
        current_ticks = ax.get_xticks()
        ax.set_xticklabels([int(tick) for tick in current_ticks])
        
        save_path = os.path.join(save_dir, f"{metric_name}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

if __name__ == "__main__":
    draw_one()