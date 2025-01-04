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
