import json

def get_metrics():
    paths = [
        "outputs/rnn/eval_outputs/metrics/df_ws20.json",
        "outputs/rnn/eval_outputs/metrics/rnn_chunk_combine_LSTM.json",
        "outputs/rnn/eval_outputs/metrics/rnn_chunk_TTT_comb_epoch3.json",
        "outputs/rnn/eval_outputs/metrics/rnn_chunk_Mamba_comb.json",
        "outputs/rnn/eval_outputs/metrics/rnn_chunk_LSTM.json",
        "outputs/rnn/eval_outputs/metrics/rnn_chunk_TTT.json",
        "outputs/rnn/eval_outputs/metrics/rnn_chunk_Mamba.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_fix_epoch1.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_fix_epoch2.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_fix_epoch3.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_fix_epoch4.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_fix_epoch5.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_pad_epoch1.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_pad_epoch2.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_pad_epoch3.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_pad_epoch4.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_pad_epoch5.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_epoch1.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_epoch2.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_epoch3.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_epoch4.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_epoch5.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_sft_epoch1.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_sft_epoch2.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_sft_epoch3.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_sft_epoch4.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_sft_epoch5.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_sft_epoch6.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_sft_unfreeze_epoch1.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_sft_unfreeze_epoch2.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_sft_unfreeze_epoch3.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_sft_unfreeze_epoch4.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_sft_unfreeze_epoch5.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_sft_unfreeze_epoch6.json",
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
    import json
    with open("data/mc_mem_data/metadata.json", 'r') as f:
        metadata = json.load(f)

    with open("data/mc_mem_data/temp.json", 'w') as f:
        json.dump(metadata["training"], f, indent=4)

if __name__ == "__main__":
    # get_metrics()
    test()