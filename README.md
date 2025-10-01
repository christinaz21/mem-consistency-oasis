# Train oasis

## Data setup

Project Url: [GitHub - openai/Video-Pre-Training: Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos](https://github.com/openai/Video-Pre-Training)

Version: 7.x

Index file url: https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_7xx_Apr_6.json

Downloading program (file path on the server):
/data/taiye/Project/open-oasis/data/VPT/download.py

Download data:

1. Download the index file
2. Set the dir_path (where to save the video and action files), the file_path (the path of index file) and start_index & end_index (download range i.e. how many data points you want to download)
3. Run the python program

## Training

1. Set `CUDA_VISIBLE_DEVICES` in main.py (if you use **ddp** as training strategy, you can just set `experiment.training.device`. But you have to set `CUDA_VISIBLE_DEVICES` if you use **deepspeed**.)
2. Set `config_path` and `config_name` in main.py
3. Download vae ckpt
4. Set config in `config/dataset/minerl.yaml`: save_dir
5. Set some config in `config/latent_diffusion.yaml`: 

    - wandb
    - algorithm.vae_ckpt
    - experiment.training.batch_size

6. python train_oasis/main.py

## Attention memory

### New Features

1. Stride: A new parameter has been introduced.
2. Difference between max_frames and n_frames:
    - max_frames refers to the number of frames that will be imported into the model.

    - n_frames refers to the number of frames loaded from the dataset.
3. Two parameter groups for the optimizer: The weight_decay parameter is set to 0 for the attention memory global weight.
4. Noise levels: Noise levels remain consistent within a chunk, whether during training or inference.
5. Gradient checkpointing: You can set gradient_checkpointing to reduce memory usage. Both `deepspeed` and `ddp` can be used.
6. bptt

### To run the code

The process is similar to running Diffusion Force. You need to configure the settings and specify the paths. Here are some important configuration details:

1. You can adjust `max_frames` and `n_frames` based on your GPU memory capacity. Ensure that `context_length` and `stride` are set to half of `max_frames`.
2. You can set the training strategy to **deepspeed**, but note that it will consume more memory.

## About the maze dataset

1. data architecture
    ```
    path.npz
        |-- image: (n, 64, 64, 3)
        |-- agent_pos: (n, 2)
        |-- agent_dir: (n,)
        |-- action: (n, 6)
        ...
    ```