import json

def get_metrics():
    paths = [
        "outputs/rnn/eval_outputs/metrics/df_ws20.json",
        "outputs/rnn/eval_outputs/metrics/rnn_chunk_combine_LSTM.json",
        "outputs/rnn/eval_outputs/metrics/rnn_chunk_TTT_comb_epoch3.json",
        "outputs/rnn/eval_outputs/metrics/rnn_chunk_Mamba_comb.json",
        "outputs/rnn/eval_outputs/metrics/rnn_chunk_LSTM.json",
        "outputs/rnn/eval_outputs/metrics/rnn_chunk_TTT.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_fix_epoch3.json",
    ]

    for path in paths:
        with open(path, 'r') as f:
            all_metrics = json.load(f)
        metrics = {}
        for item in all_metrics:
            prompt_frames = item["prompt_frames"]
            for key, values in item["metrics"].items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].extend(values[prompt_frames:])
        avg_metrics = {key: sum(values) / len(values) for key, values in metrics.items()}
        print(f"Metrics for {path}:")
        for key, value in avg_metrics.items():
            print(f"{key}: {value:.4f}")
        print()

def test():
    import torch
    from einops import rearrange
    a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    a_repeat = a.repeat(2, 1)
    a = a.unsqueeze(1).expand(-1, 2, -1)
    a = rearrange(a, "B T D -> (T B) D")
    print(torch.equal(a, a_repeat))

if __name__ == "__main__":
    get_metrics()
    # test()