from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import torch

def convert():
    path = "/data/taiye/Project/train-oasis/outputs/2025-02-15/12-48-53/checkpoints/epoch=0-step=36000.ckpt"
    ckpt = get_fp32_state_dict_from_zero_checkpoint(path)
    # state_dict = {}
    # for key, value in ckpt.items():
    #     print(key)
    #     if key.startswith("diffusion_model."):
    #         state_dict[key[16:]] = value
    output_path = "/data/taiye/Project/train-oasis/outputs/2025-02-15/12-48-53/checkpoints/window_size=30-step=36000.bin"
    # model.load_state_dict(state_dict, strict=True)
    torch.save(ckpt, output_path)

def check():
    path = "/data/taiye/Project/train-oasis/outputs/2025-02-15/12-48-53/checkpoints/epoch=0-step=12000.bin"
    state_dict = torch.load(path)
    for key, value in state_dict.items():
        print(key)

if __name__ == "__main__":
    convert()
    # check()