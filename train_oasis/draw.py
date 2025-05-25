import matplotlib.pyplot as plt
import numpy as np
import json
import os
import cv2
import seaborn as sns
import wandb

def draw_all():
    models = ["vanilla_10", "vanilla_20", "world_coordinate", "infini_attn", "rag", "yarn", "historical_buffer", "rag_wo_training", "rag_multi"]
    metrics = ["mse", "psnr", "ssim", "lpips", "1000_step_resnet"]
    splits = ["memory", "random"]
    save_dir = "/home/tc0786/Project/train-oasis/outputs/eval_outputs/figures"

    for split in splits:
        if split == "memory":
            limit = 300
        elif split == "random":
            limit = 1200
        else:
            raise ValueError(f"Unknown split: {split}")
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            for model in models:
                json_path = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model}/{split}.json"
                with open(json_path, "r") as f:
                    data = json.load(f)
                values = []
                for item in data:
                    values.append(item["output_dict"][metric][:limit])
                values = np.array(values) # (b, t)
                values = values.mean(axis=0)
                plt.plot(values, label=model)
            plt.rcParams["font.size"] = 15
            plt.title(f"{metric} for {split} split")
            plt.xlabel("Sample Index")
            plt.ylabel(metric)
            plt.legend()
            plt.grid()
            save_path = os.path.join(save_dir, split, f"{metric}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

def draw_one():
    # models = ["vanilla_10", "vanilla_20", "world_coordinate", "infini_attn", "rag", "yarn", "historical_buffer", "rag_wo_training"]
    models = ["vanilla_10", "vanilla_20", "rag", "yarn", "historical_buffer", "infini_attn"]
    model_names = ["DF(window 10)", "DF(window 20)", "VRAG", "YaRN", "History Buffer", "infini_attn"]
    metrics = ["ssim"]
    splits = ["memory"] # ["memory", "random"]
    save_dir = "/home/tc0786/Project/train-oasis/outputs/eval_outputs/figures/world_consistency"

    for split in splits:
        if split == "memory":
            limit = 300
        elif split == "random":
            limit = 1200
        else:
            raise ValueError(f"Unknown split: {split}")
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            for model, model_name in zip(models, model_names):
                json_path = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model}/{split}.json"
                with open(json_path, "r") as f:
                    data = json.load(f)
                values = []
                for item in data:
                    values.append(item["output_dict"][metric][100:limit])
                values = np.array(values) # (b, t)
                values = values.mean(axis=0)
                plt.plot(values, label=model_name)
            plt.rcParams["font.size"] = 15
            plt.title(f"{metric} for World Consistency")
            plt.xlabel("Sample Index")
            plt.ylabel(metric)
            plt.legend()
            plt.grid()
            save_path = os.path.join(save_dir, f"{metric}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

def print_table():
    models = ["rag", "rag_wo_training", "world_coordinate"] # ["vanilla_10", "vanilla_20", "world_coordinate", "infini_attn", "rag", "yarn", "historical_buffer", "rag_wo_training"]
    metrics = ["ssim", "psnr", "lpips"] # ["mse", "psnr", "ssim", "uiqi", "lpips", "fid", "full_step_resnet", "small_dit", "1000_step_resnet"]
    splits = ["memory"] # ["memory", "random"]

    for split in splits:
        print(f"Split: {split}")
        if split == "memory":
            limit = 300
        elif split == "random":
            limit = 1200
        else:
            raise ValueError(f"Unknown split: {split}")
        for model in models:
            print(model)
            json_path = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model}/{split}.json"
            with open(json_path, "r") as f:
                data = json.load(f)
            for metric in metrics:
                values = []
                for item in data:
                    item = item["output_dict"][metric]
                    if isinstance(item, list):
                        values.append(item[:limit])
                    elif isinstance(item, float):
                        values.append(item)
                    else:
                        raise ValueError(f"Unknown metric type: {type(item)}")
                values = np.array(values)
                values = values.mean()
                print(f"{values:.3f}", end=" & ")
            print()

def draw_ablation_predx():
    models = ["vanilla_20", "pred_x"]
    metrics = ["mse", "psnr", "ssim", "lpips", "1000_step_resnet"]
    splits = ["memory", "random"]
    save_dir = "/home/tc0786/Project/train-oasis/outputs/eval_outputs/figures/ablation/pred_x"

    for split in splits:
        if split == "memory":
            limit = 300
        elif split == "random":
            limit = 1200
        else:
            raise ValueError(f"Unknown split: {split}")
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            for model in models:
                json_path = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model}/{split}.json"
                with open(json_path, "r") as f:
                    data = json.load(f)
                values = []
                for item in data:
                    values.append(item["output_dict"][metric][:limit])
                values = np.array(values) # (b, t)
                values = values.mean(axis=0)
                plt.plot(values, label=model)
            plt.rcParams["font.size"] = 15
            plt.title(f"{metric} for {split} split")
            plt.xlabel("Sample Index")
            plt.ylabel(metric)
            plt.legend()
            plt.grid()
            save_path = os.path.join(save_dir, split, f"{metric}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

def draw_ablation_rag_training():
    models = ["rag", "rag_wo_training", "world_coordinate"]
    model_names = ["VRAG", "VRAG (no training)", "VRAG (no memory)"]
    metrics = ["ssim"]
    splits = ["memory", "random"]
    save_dir = "/home/tc0786/Project/train-oasis/outputs/eval_outputs/figures/ablation/rag_training"

    for split in splits:
        if split == "memory":
            limit = 300
            title = "World Consistency"
        elif split == "random":
            limit = 1200
            title = "Compounding Error"
        else:
            raise ValueError(f"Unknown split: {split}")
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            for model, model_name in zip(models, model_names):
                json_path = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model}/{split}.json"
                with open(json_path, "r") as f:
                    data = json.load(f)
                values = []
                for item in data:
                    values.append(item["output_dict"][metric][100:limit])
                values = np.array(values) # (b, t)
                values = values.mean(axis=0)
                plt.plot(values, label=model_name)
            plt.rcParams["font.size"] = 15
            plt.title(f"{metric} for {title}")
            plt.xlabel("Sample Index")
            plt.ylabel(metric)
            plt.legend()
            plt.grid()
            save_path = os.path.join(save_dir, split, f"{metric}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

def draw_pic_memory():
    path = "/home/tc0786/Project/train-oasis/data/eval_data/paths.json"
    with open(path, "r") as f:
        data = json.load(f)["memory"]
    for d in data:
        relative_path = d["save_relative_path"]
        gt_video = f"/home/tc0786/Project/train-oasis/data/eval_data/{relative_path}"
        save_file_name = relative_path.replace("mp4", "png")
        print(save_file_name)
        save_path = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/figures/real_image/{save_file_name}"

        model_names = ["vanilla_10", "vanilla_20", "infini_attn", "yarn", "historical_buffer", "rag"]
        model_labels = ["DF10", "DF20", "Neur", "YaRN", "Buffer", "VRAG"]

        skip = 4
        frame_idxs = [99+i*skip for i in range(6)]
        
        fig, axes = plt.subplots(len(model_names) + 1, 6, figsize=(30, 20))

        for model_idx, (model_name, model_label) in enumerate(zip(model_names, model_labels)):
            target_video = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model_name}/{relative_path}"
            cap = cv2.VideoCapture(target_video)
            frames = []
            for idx in frame_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    raise ValueError(f"Failed to read frame {idx} from {target_video}")
            cap.release()

            for i, frame in enumerate(frames):
                axes[model_idx, i].imshow(frame)
                # 先设置ylabel再关闭轴
                if i == 0:
                    axes[model_idx, i].set_ylabel(model_label, fontsize=20)
                axes[model_idx, i].set_xticks([])
                axes[model_idx, i].set_yticks([])
                axes[model_idx, i].set_frame_on(False)  # 替代axis("off")

        # Process ground truth video
        cap = cv2.VideoCapture(gt_video)
        gt_frames = []
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gt_frames.append(frame)
            else:
                raise ValueError(f"Failed to read frame {idx} from {gt_video}")
        cap.release()

        for i, frame in enumerate(gt_frames):
            axes[-1, i].imshow(frame)
            if i == 0:
                axes[-1, i].set_ylabel("GT", fontsize=20)
            axes[-1, i].set_xticks([])
            axes[-1, i].set_yticks([])
            axes[-1, i].set_frame_on(False)

        plt.tight_layout()
        fig.subplots_adjust(left=0.1, wspace=0.02, hspace=0.02)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.clf()

def draw_pic_random():
    for frame_idx in range(20):
        gt_video = f"/home/tc0786/Project/train-oasis/data/eval_data/random/{frame_idx:06d}.mp4"
        save_path = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/figures/real_image/random/{frame_idx:06d}.png"

        model_names = ["vanilla_10", "vanilla_20", "infini_attn", "yarn", "historical_buffer", "rag"]
        model_labels = ["DF10", "DF20", "Neur", "YaRN", "Buffer", "VRAG"]

        skip = 200
        frame_idxs = [99+i*skip for i in range(6)]
        
        fig, axes = plt.subplots(len(model_names) + 1, 6, figsize=(30, 20))

        for model_idx, (model_name, model_label) in enumerate(zip(model_names, model_labels)):
            target_video = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model_name}/random/{frame_idx:06d}.mp4"
            cap = cv2.VideoCapture(target_video)
            frames = []
            for idx in frame_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    raise ValueError(f"Failed to read frame {idx} from {target_video}")
            cap.release()

            for i, frame in enumerate(frames):
                axes[model_idx, i].imshow(frame)
                # 先设置ylabel再关闭轴
                if i == 0:
                    axes[model_idx, i].set_ylabel(model_label, fontsize=20)
                axes[model_idx, i].set_xticks([])
                axes[model_idx, i].set_yticks([])
                axes[model_idx, i].set_frame_on(False)  # 替代axis("off")

        # Process ground truth video
        cap = cv2.VideoCapture(gt_video)
        gt_frames = []
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gt_frames.append(frame)
            else:
                raise ValueError(f"Failed to read frame {idx} from {gt_video}")
        cap.release()

        for i, frame in enumerate(gt_frames):
            axes[-1, i].imshow(frame)
            if i == 0:
                axes[-1, i].set_ylabel("GT", fontsize=20)
            axes[-1, i].set_xticks([])
            axes[-1, i].set_yticks([])
            axes[-1, i].set_frame_on(False)

        plt.tight_layout()
        fig.subplots_adjust(left=0.1, wspace=0.02, hspace=0.02)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.clf()

def draw_collapse():
    metrics = ["ssim", "psnr", "lpips", "1000_step_resnet"]
    metrics_name = ["SSIM", "PSNR", "LPIPS", "Discriminator"]
    selected_idx = [0, 270, 540, 810, 1080]
    # path = "/home/tc0786/Project/train-oasis/outputs/eval_outputs/vanilla_20/random/000002.mp4"
    result_path = "/home/tc0786/Project/train-oasis/outputs/eval_outputs/vanilla_20/random.json"
    with open(result_path, "r") as f:
        data = json.load(f)

    plt.rcParams["font.size"] = 24
    plt.figure(figsize=(30, 10))
    def smooth(x, half_window=5):
        l = len(x)
        new_x = np.zeros(l)
        for i in range(l):
            clip = x[max(0, i - half_window):min(l, i + half_window + 1)]
            new_x[i] = np.mean(clip)
        return new_x
    
    for metric, metric_name in zip(metrics, metrics_name):
        values = data[2]["output_dict"][metric]
        values = np.array(values)
        if metric == "lpips":
            values = - values
        if metric == "small_dit":
            print(len(values))
        values = (values - values.min()) / (values.max() - values.min())
        values = smooth(values, 10)
        plt.plot(values, label=metric_name)
    for idx in selected_idx:
        plt.axvline(x=idx, color='k', linestyle='--', linewidth=3)
    plt.title("Metrics", fontsize=30)
    plt.xlabel("Sample Index", fontsize=30)
    plt.xlim(-10, 1090)
    plt.ylabel("Metric Value", fontsize=30)
    plt.legend()
    plt.grid()
    save_path = "/home/tc0786/Project/train-oasis/outputs/eval_outputs/figures/collapse/metric.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def draw_collapse_pic():
    path = "/home/tc0786/Project/train-oasis/outputs/eval_outputs/vanilla_20/random/000002.mp4"
    frame_idxs = [0, 270, 540, 810, 1080]

    cap = cv2.VideoCapture(path)
    frames = []
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            raise ValueError(f"Failed to read frame {idx} from {path}")
    cap.release()
    fig, axes = plt.subplots(1, len(frame_idxs), figsize=(30, 10))
    for i, frame in enumerate(frames):
        axes[i].imshow(frame)
        axes[i].set_xlabel(f"Frame {frame_idxs[i]}", fontsize=24)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_frame_on(False)
    fig.subplots_adjust(wspace=0.02)

    save_path = "/home/tc0786/Project/train-oasis/outputs/eval_outputs/figures/collapse/real.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def find_similar():
    from torchvision.io import read_video
    from fractions import Fraction
    from torchmetrics.functional import structural_similarity_index_measure
    import torch
    from tqdm import tqdm
    # dir1 = "/home/tc0786/Project/train-oasis/data/eval_data/additional_mem"
    # dir2 = "/home/tc0786/Project/train-oasis/outputs/eval_outputs/additional_mem_rag"
    dir1 = "/home/tc0786/Project/train-oasis/data/eval_data/additional_rotate"
    dir2 = "/home/tc0786/Project/train-oasis/outputs/eval_outputs/rag/memory/additional_rotate"

    length = 15
    start = Fraction(100, 20)
    end = Fraction(100 + length, 20)

    all_ssim = []
    for i in tqdm(range(0, 600)):
        video_file = f"{i:06d}.mp4"
        video_path1 = os.path.join(dir1, video_file)
        video_path2 = os.path.join(dir2, video_file)
        if not os.path.exists(video_path1) or not os.path.exists(video_path2):
            print(f"File not found: {video_path1} or {video_path2}")
            continue
        try:
            video1, _, _ = read_video(video_path1, start_pts=start, end_pts=end, pts_unit="sec")
            video2, _, _ = read_video(video_path2, start_pts=start, end_pts=end, pts_unit="sec")
            video1 = video1.contiguous().numpy()
            video2 = video2.contiguous().numpy()
            video1 = torch.from_numpy(video1).float() / 255.0
            video2 = torch.from_numpy(video2).float() / 255.0
            video1 = video1.permute(0, 3, 1, 2)
            video2 = video2.permute(0, 3, 1, 2)
            ssim = structural_similarity_index_measure(video2, video1, data_range=2.0).cpu().tolist()
            all_ssim.append((video_file, ssim))
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            continue
    # Sort and print the top 5 videos with the highest SSIM
    top_20 = sorted(all_ssim, key=lambda x: x[1], reverse=True)[:20]
    print("\nTop 5 videos with highest SSIM:")
    for video, ssim in top_20:
        print(f"{video}: {ssim:.4f}")

def find_similar_2():
    from torchvision.io import read_video
    from fractions import Fraction
    from torchmetrics.functional import structural_similarity_index_measure
    import torch

    path = "/home/tc0786/Project/train-oasis/data/eval_data/paths.json"
    save_dir = "/home/tc0786/Project/train-oasis/outputs/eval_outputs/rag"
    with open(path, "r") as f:
        data = json.load(f)

    length = 30
    start = Fraction(100, 20)
    end = Fraction(100 + length, 20)

    all_ssim = []
    for d in data["memory"]:
        gt_path = d["video_path"]
        save_relative_path = d["save_relative_path"]
        video_path = os.path.join(save_dir, save_relative_path)
        video_gt, _, _ = read_video(gt_path, start_pts=start, end_pts=end, pts_unit="sec")
        video_pred, _, _ = read_video(video_path, start_pts=start, end_pts=end, pts_unit="sec")
        video_gt = video_gt.contiguous().numpy()
        video_pred = video_pred.contiguous().numpy()
        video_gt = torch.from_numpy(video_gt).float() / 255.0
        video_pred = torch.from_numpy(video_pred).float() / 255.0
        video_gt = video_gt.permute(0, 3, 1, 2)
        video_pred = video_pred.permute(0, 3, 1, 2)
        ssim = structural_similarity_index_measure(video_pred, video_gt, data_range=2.0).cpu().tolist()
        print(save_relative_path, ssim)
        all_ssim.append((save_relative_path, ssim))

    # Sort and print the top 5 videos with the highest SSIM
    top_5 = sorted(all_ssim, key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 videos with highest SSIM:")
    for video, ssim in top_5:
        print(f"{video}: {ssim:.4f}")

def draw_one_fig():
    import matplotlib.pyplot as plt

    relative_paths = [
        "memory/rotate/000006.mp4",
        "memory/hex/000008.mp4",
        "memory/rotate_wait/000001.mp4",
        "memory/straight/000000.mp4",
    ]

    for idx, relative_path in enumerate(relative_paths):
        gt_video = f"/home/tc0786/Project/train-oasis/data/eval_data/{relative_path}"
        save_file_name = relative_path.replace("mp4", "png")
        print(save_file_name)
        save_path = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/figures/real_image/selected/{idx:06d}.png"

        model_names = ["vanilla_20", "infini_attn", "yarn", "historical_buffer", "rag"]
        model_labels = ["DF20", "Neural Memory", "YaRN", "History Buffer", "VRAG"]

        skip = 5
        frame_idxs = [100+i*skip for i in range(4)]
        
        fig, axes = plt.subplots(len(model_names) + 1, 4, figsize=(25, 20))

        for model_idx, (model_name, model_label) in enumerate(zip(model_names, model_labels)):
            target_video = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model_name}/{relative_path}"
            cap = cv2.VideoCapture(target_video)
            frames = []
            for idx in frame_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    raise ValueError(f"Failed to read frame {idx} from {target_video}")
            cap.release()

            for i, frame in enumerate(frames):
                axes[model_idx, i].imshow(frame)
                # 先设置ylabel再关闭轴
                if i == 0:
                    axes[model_idx, i].set_ylabel(model_label, fontsize=20)
                axes[model_idx, i].set_xticks([])
                axes[model_idx, i].set_yticks([])
                axes[model_idx, i].set_frame_on(False)  # 替代axis("off")

        # Process ground truth video
        cap = cv2.VideoCapture(gt_video)
        gt_frames = []
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gt_frames.append(frame)
            else:
                raise ValueError(f"Failed to read frame {idx} from {gt_video}")
        cap.release()

        for i, frame in enumerate(gt_frames):
            axes[-1, i].imshow(frame)
            if i == 0:
                axes[-1, i].set_ylabel("GT", fontsize=20)
            axes[-1, i].set_xlabel(f"Frame {frame_idxs[i]}", fontsize=20)
            axes[-1, i].set_xticks([])
            axes[-1, i].set_yticks([])
            axes[-1, i].set_frame_on(False)

        plt.tight_layout()
        fig.subplots_adjust(left=0.1, wspace=0.02, hspace=0.02)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.clf()

def draw_select_random():
    relative_paths = [
        "random/000009.mp4",
        "random/000017.mp4",
    ]

    for idx, relative_path in enumerate(relative_paths):
        save_file_name = relative_path.replace("mp4", "png")
        print(save_file_name)
        save_path = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/figures/real_image/selected/{idx:06d}.png"

        model_names = ["vanilla_20", "yarn", "historical_buffer", "rag"]
        model_labels = ["DF20", "YaRN", "History Buffer", "VRAG"]

        skip = 300
        frame_idxs = [200+i*skip for i in range(4)]
        
        fig, axes = plt.subplots(len(model_names), 4, figsize=(25, 13))

        for model_idx, (model_name, model_label) in enumerate(zip(model_names, model_labels)):
            target_video = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model_name}/{relative_path}"
            cap = cv2.VideoCapture(target_video)
            frames = []
            for idx in frame_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    raise ValueError(f"Failed to read frame {idx} from {target_video}")
            cap.release()

            for i, frame in enumerate(frames):
                # 增加明度
                if model_name == "rag":
                    # frame = np.clip(frame * 1.3, 0, 255).astype(np.uint8)
                    frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=30)
                axes[model_idx, i].imshow(frame)
                # 先设置ylabel再关闭轴
                if i == 0:
                    axes[model_idx, i].set_ylabel(model_label, fontsize=24)
                if model_idx == (len(model_names) - 1):
                    axes[model_idx, i].set_xlabel(f"Frame {frame_idxs[i]}", fontsize=24)
                axes[model_idx, i].set_xticks([])
                axes[model_idx, i].set_yticks([])
                axes[model_idx, i].set_frame_on(False)  # 替代axis("off")

        plt.tight_layout()
        fig.subplots_adjust(left=0.1, wspace=0.02, hspace=0.02)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.clf()

def draw_select():
    relative_paths = [
        "memory/rotate/000006.mp4",
        "memory/hex/000008.mp4",
        "memory/rotate_wait/000001.mp4",
        "memory/straight/000000.mp4",
    ]

    for idx, relative_path in enumerate(relative_paths):
        save_file_name = relative_path.replace("mp4", "png")
        print(save_file_name)
        gt_video = f"/home/tc0786/Project/train-oasis/data/eval_data/{relative_path}"
        save_path = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/figures/real_image/selected/{idx:06d}.png"

        model_names = ["vanilla_20", "yarn", "infini_attn", "historical_buffer", "rag"]
        model_labels = ["DF20", "YaRN", "Neural Memory", "History Buffer", "VRAG"]

        skip = 4
        frame_idxs = [103+i*skip for i in range(4)]
        
        fig, axes = plt.subplots(len(model_names) + 1, 4, figsize=(24, 18))

        for model_idx, (model_name, model_label) in enumerate(zip(model_names, model_labels)):
            target_video = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model_name}/{relative_path}"
            cap = cv2.VideoCapture(target_video)
            frames = []
            for idx in frame_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    raise ValueError(f"Failed to read frame {idx} from {target_video}")
            cap.release()

            for i, frame in enumerate(frames):
                axes[model_idx, i].imshow(frame)
                # 先设置ylabel再关闭轴
                if i == 0:
                    axes[model_idx, i].set_ylabel(model_label, fontsize=24)
                axes[model_idx, i].set_xticks([])
                axes[model_idx, i].set_yticks([])
                axes[model_idx, i].set_frame_on(False)  # 替代axis("off")

        # Process ground truth video
        cap = cv2.VideoCapture(gt_video)
        gt_frames = []
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gt_frames.append(frame)
            else:
                raise ValueError(f"Failed to read frame {idx} from {gt_video}")
        cap.release()

        for i, frame in enumerate(gt_frames):
            axes[-1, i].imshow(frame)
            if i == 0:
                axes[-1, i].set_ylabel("GT", fontsize=24)
            axes[-1, i].set_xticks([])
            axes[-1, i].set_yticks([])
            axes[-1, i].set_frame_on(False)

        plt.tight_layout()
        fig.subplots_adjust(left=0.1, wspace=0.02, hspace=0.02)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.clf()

def draw_ablation_yarn():
    sns.set_theme(style="whitegrid")  # Set Seaborn theme

    models = ["yarn", "vanilla_40_direct_extrapolate"]
    model_names = ["YaRN", "DF(window 40)"]
    # models = ["vanilla_10", "vanilla_20", "rag", "yarn", "historical_buffer"]
    # model_names = ["DF(window 10)", "DF(window 20)", "VRAG", "YaRN", "History Buffer"]
    metrics = ["ssim", "lpips", "psnr"]
    splits = ["memory", "random"]
    save_dir = "/scratch/gpfs/CHIJ/taiye/outputs/figures/ablation/yarn"
    # save_dir = "/scratch/gpfs/CHIJ/taiye/outputs/draw_figs/compounding_error"

    for split in splits:
        if split == "memory":
            limit = 300
            smoothing_window = 2   # 2 for world coherence, 10 for compounding error
 
        elif split == "random":
            limit = 1200
            smoothing_window = 10
        else:
            raise ValueError(f"Unknown split: {split}")

        for metric in metrics:
            plt.figure(figsize=(10, 4))
            for model, model_name in zip(models, model_names):
                json_path = f"/scratch/gpfs/CHIJ/taiye/outputs/{model}/{split}.json"
                with open(json_path, "r") as f:
                    data = json.load(f)
                values = []
                for item in data:
                    values.append(item["output_dict"][metric][100:limit])
                values = np.array(values)  # (b, t)
                values = values.mean(axis=0)
                # Apply smoothing with configurable window size
                smoothed_values = np.convolve(values, np.ones(smoothing_window)/smoothing_window, mode='valid')
                sns.lineplot(x=np.arange(len(smoothed_values)), y=smoothed_values, label=model_name, linewidth=3.5)

            # plt.title(f"World Coherence ({metric.upper()})", fontsize=18)
            plt.xlabel("Frame Index", fontsize=24)
            plt.ylabel(metric.upper(), fontsize=24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            # plt.legend(fontsize=2)
            plt.legend(fontsize=14, loc="lower right")
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
            
            # Set x-axis to start from 0 and offset ticks by 100
            plt.xlim(left=0)
            current_ticks = ax.get_xticks()
            ax.set_xticklabels([int(tick + 100) for tick in current_ticks])
            
            save_path = os.path.join(save_dir, f"{split}_{metric}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()

def draw_ablation_yarn_pic():
    relative_paths = [f"random/{i:06d}.mp4" for i in range(20)]

    for idx, relative_path in enumerate(relative_paths):
        save_file_name = relative_path.replace("mp4", "png")
        print(save_file_name)
        gt_video = f"/home/tc0786/Project/train-oasis/data/eval_data/{relative_path}"
        save_path = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/figures/ablation/yarn_real_image/{idx:06d}.png"

        model_names = ["yarn", "vanilla_40_direct_extrapolate"]
        model_labels = ["YaRN", "DF(window 40)"]

        skip = 300
        frame_idxs = [100+i*skip for i in range(4)]
        
        fig, axes = plt.subplots(len(model_names) + 1, 4, figsize=(24, 10))

        for model_idx, (model_name, model_label) in enumerate(zip(model_names, model_labels)):
            target_video = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model_name}/{relative_path}"
            cap = cv2.VideoCapture(target_video)
            frames = []
            for idx in frame_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    raise ValueError(f"Failed to read frame {idx} from {target_video}")
            cap.release()

            for i, frame in enumerate(frames):
                axes[model_idx, i].imshow(frame)
                # 先设置ylabel再关闭轴
                if i == 0:
                    axes[model_idx, i].set_ylabel(model_label, fontsize=24)
                axes[model_idx, i].set_xticks([])
                axes[model_idx, i].set_yticks([])
                axes[model_idx, i].set_frame_on(False)  # 替代axis("off")

        # Process ground truth video
        cap = cv2.VideoCapture(gt_video)
        gt_frames = []
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gt_frames.append(frame)
            else:
                raise ValueError(f"Failed to read frame {idx} from {gt_video}")
        cap.release()

        for i, frame in enumerate(gt_frames):
            axes[-1, i].imshow(frame)
            if i == 0:
                axes[-1, i].set_ylabel("GT", fontsize=24)
            axes[-1, i].set_xlabel(f"Frame {frame_idxs[i]}", fontsize=24)
            axes[-1, i].set_xticks([])
            axes[-1, i].set_yticks([])
            axes[-1, i].set_frame_on(False)

        plt.tight_layout()
        fig.subplots_adjust(left=0.1, wspace=0.02, hspace=0.02)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.clf()

def draw_loss_curve():
    # 你需要填写这三个run id
    run_ids = [
        "v1kv0n5a",
        "29sgfy5t",
        "h01pmba7",
    ]
    run_names = [
        "DF(window 20)",
        "VRAG (no memory)",
        "Neural Memory",
    ]
    project = "appendix"

    metric = "training/loss" # 你要画的loss曲线名

    plt.figure(figsize=(10, 6))
    for run_id, run_name in zip(run_ids, run_names):
        api = wandb.Api()
        run = api.run(f"{project}/{run_id}")
        history = run.history(samples=10000)  # 下载所有step的历史
        values = history[metric]
        indices = values.index[~values.isna()]
        values = values.dropna().values
        plt.plot(indices, values, label=run_name)

    plt.xlim(-1, 1201)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid()
    save_path = "/home/tc0786/Project/train-oasis/outputs/eval_outputs/figures/loss_curve.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    draw_all()