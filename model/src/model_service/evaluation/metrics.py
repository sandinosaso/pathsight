import json
import datetime
from model_service.config import ModelServiceConfig

def calculate_and_save_metrics(history):
    config = ModelServiceConfig()

    # 1. Create unique timestamp for this specific run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    # 2. Extract best metrics (at the point where EarlyStopping restored weights)
    monitor = config.train.early_stop_monitor
    if config.train.early_stop_mode == 'max':
        best_idx = history.history[monitor].index(max(history.history[monitor]))
    else:
        best_idx = history.history[monitor].index(min(history.history[monitor]))

    final_metrics = {m: float(v[best_idx]) for m, v in history.history.items()}

    # 3. Calculate F1-Score manually
    p = final_metrics.get('val_precision', 0)
    r = final_metrics.get('val_recall', 0)
    final_metrics['val_f1_score'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    final_metrics['run_timestamp'] = timestamp

    # 4. Save to artifacts/metrics/
    json_path = config.paths.artifacts_metrics / f"metrics_{timestamp}.json"
    config.paths.artifacts_metrics.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"✅ Training Complete. Metrics saved to {json_path}")

    return final_metrics
