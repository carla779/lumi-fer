import json

path = r"models_cnn_raf7\raf7_csv_thresholds.json"
d = json.load(open(path, "r", encoding="utf-8"))

print("precision_target:", d["precision_target"])
print("min_pred_pos:", d["min_pred_pos"])
print("\nBest thetas (precision-first if available, else F0.5 fallback):\n")

for emo, info in d["thresholds"].items():
    t = info["theta_precision"] or info["theta_f05"]
    print(f"{emo:9s}  theta={t['theta']:.3f}  prec={t['prec']:.3f}  rec={t['rec']:.3f}  pos={t['pos']}")