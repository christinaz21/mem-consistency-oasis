import json

def get_metrics():
    paths = [
        "outputs/rnn/eval_outputs/metrics/df_ws20.json",
        "outputs/rnn/eval_outputs/metrics/rnn_chunk_combine_LSTM.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_b256_epoch3.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_b256_epoch2.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_b256_epoch1.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_pad_epoch3.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_epoch1.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_epoch2.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_epoch3.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_epoch4.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_epoch5.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_epoch6.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_aux_epoch3.json",
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

if __name__ == "__main__":
    get_metrics()