import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Data ---

mobilenet_data = """Epoch,TrainLoss,TrainAccuracy,ValLoss,ValAccuracy
1,0.5046,79.36,8.9580,37.04
2,0.5576,78.12,9.0318,37.08
3,0.3787,84.76,3.9157,54.96
4,0.3045,89.06,3.5121,55.24
5,0.3470,86.73,7.5405,56.65
6,0.2398,93.75,7.9950,55.60
7,0.2386,90.50,5.4689,51.95
8,0.3667,85.94,5.3532,52.03
9,0.1620,93.68,4.0561,55.16
10,0.0838,96.88,4.0124,55.20
11,0.1459,94.38,2.3715,65.45
12,0.0419,100.00,2.3650,65.49
13,0.1370,94.94,1.5627,72.84
14,0.0675,95.31,1.5599,72.92
15,0.1274,94.78,1.2278,77.34
16,0.1045,95.31,1.2242,77.38
17,0.1185,95.38,0.9356,80.80
18,0.1652,92.19,0.9341,80.92
19,0.1185,95.26,0.6905,83.41
20,0.1293,92.19,0.6881,83.41
21,0.1082,95.92,0.5421,86.06
22,0.1575,93.75,0.5415,86.06
23,0.0999,96.01,0.4496,87.59
24,0.0979,95.31,0.4482,87.55
25,0.0947,96.48,0.3918,88.11
26,0.0208,100.00,0.3918,88.07
27,0.0881,96.82,0.3559,89.03
28,0.1051,93.75,0.3564,89.07
29,0.0861,96.65,0.3395,89.39
30,0.0924,95.31,0.3397,89.35
31,0.0825,96.86,0.3246,89.51
32,0.0805,95.31,0.3247,89.43
33,0.0753,97.20,0.3237,89.27
34,0.1270,95.31,0.3219,89.35
35,0.0722,97.35,0.3337,89.03
36,0.0439,98.44,0.3340,88.99
37,0.0608,97.72,0.3236,89.67
38,0.0354,98.44,0.3240,89.67
39,0.0619,97.76,0.3244,89.55
40,0.0456,98.44,0.3244,89.55
41,0.0599,97.77,0.3206,89.63
42,0.0761,95.31,0.3205,89.63
43,0.0586,97.85,0.3179,90.04
44,0.0956,93.75,0.3179,90.04
45,0.0588,97.88,0.3165,90.16
46,0.0527,98.44,0.3166,90.16
47,0.0599,97.71,0.3170,90.00
48,0.1301,96.88,0.3170,90.00
49,0.0591,97.85,0.3158,90.12
50,0.0256,100.00,0.3158,90.12"""

dinov2_data = """Epoch,TrainLoss,TrainAccuracy,ValLoss,ValAccuracy
1,0.6447,71.55,0.4842,79.83
2,0.4432,81.73,0.4208,82.00
3,0.3959,83.67,0.3939,83.85
4,0.3711,85.02,0.3784,84.17
5,0.3545,85.68,0.3855,83.69
6,0.3417,86.15,0.3588,84.89
7,0.3320,86.32,0.3625,84.65
8,0.3304,86.35,0.3479,85.70
9,0.3200,87.10,0.3621,84.41
10,0.3147,87.16,0.3387,86.06
11,0.3043,87.62,0.3531,85.38
12,0.3037,87.88,0.3328,86.34
13,0.2990,87.77,0.3324,86.18
14,0.2940,88.01,0.3421,85.94
15,0.2942,87.92,0.3288,86.26
16,0.2906,88.10,0.3294,86.30
17,0.2823,88.54,0.3304,86.34
18,0.2826,88.44,0.3276,86.46
19,0.2813,88.47,0.3258,86.38
20,0.2821,88.76,0.3255,86.70
21,0.2736,88.87,0.3405,85.09
22,0.2772,88.60,0.3273,86.50
23,0.2730,89.08,0.3289,86.50
24,0.2715,89.13,0.3648,84.73
25,0.2696,89.00,0.3348,86.26
26,0.2643,89.25,0.3394,85.94
27,0.2642,89.44,0.3293,86.54
28,0.2636,89.45,0.3296,86.26
29,0.2649,89.30,0.3339,86.42
30,0.2632,89.52,0.3287,86.62
31,0.2697,89.25,0.3516,85.62
32,0.2590,89.68,0.3265,86.42
33,0.2567,89.95,0.3314,86.46
34,0.2594,89.85,0.3341,86.26
35,0.2629,89.30,0.3288,86.82
36,0.2534,89.88,0.3344,86.46
37,0.2540,89.94,0.3487,85.78
38,0.2599,89.59,0.3378,86.06
39,0.2557,89.89,0.3489,86.10
40,0.2520,89.82,0.3444,85.98
41,0.2508,90.15,0.3778,85.09
42,0.2553,89.70,0.3371,86.50
43,0.2463,89.97,0.3367,86.38
44,0.2470,90.40,0.3336,86.34
45,0.2468,90.30,0.3476,86.14
46,0.2533,90.01,0.3361,86.34
47,0.2485,90.07,0.3319,86.58
48,0.2417,90.45,0.3345,86.18
49,0.2444,90.23,0.3791,85.21
50,0.2473,90.15,0.3402,85.94"""

# --- Directories ---
os.makedirs("results", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# --- CSV Generation ---
def parse_and_save_csv(raw_data, filepath):
    lines = raw_data.strip().split('\n')
    header = lines[0].split(',')
    # Transform header to desired format
    header = ["Epoch", "Train_Loss", "Train_Accuracy", "Val_Loss", "Val_Accuracy"]
    
    with open(filepath, 'w') as f:
        f.write(','.join(header) + '\n')
        for line in lines[1:]:
            f.write(line + '\n')

parse_and_save_csv(mobilenet_data, "results/mobilenetv2_epoch_history.csv")
parse_and_save_csv(dinov2_data, "results/dinov2_epoch_history.csv")

# --- Plotting ---
df_mobile = pd.read_csv("results/mobilenetv2_epoch_history.csv")
df_dino = pd.read_csv("results/dinov2_epoch_history.csv")

def plot_graph(df, title, y_label, columns, filename):
    plt.figure(figsize=(10, 6))
    for col, label in columns:
        plt.plot(df["Epoch"], df[col], label=label, marker='o', markersize=4)
    
    plt.title(title, fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=500, bbox_inches="tight")
    plt.close()

# MobileNetV2 Accuracy
plot_graph(
    df=df_mobile,
    title="MobileNetV2 Training and Validation Accuracy",
    y_label="Accuracy (%)",
    columns=[("Train_Accuracy", "Training Accuracy"), ("Val_Accuracy", "Validation Accuracy")],
    filename="graphs/mobilenetv2_accuracy.png"
)

# MobileNetV2 Loss
plot_graph(
    df=df_mobile,
    title="MobileNetV2 Training and Validation Loss",
    y_label="Loss",
    columns=[("Train_Loss", "Training Loss"), ("Val_Loss", "Validation Loss")],
    filename="graphs/mobilenetv2_loss.png"
)

# DINOv2 Accuracy
plot_graph(
    df=df_dino,
    title="DINOv2 Training and Validation Accuracy",
    y_label="Accuracy (%)",
    columns=[("Train_Accuracy", "Training Accuracy"), ("Val_Accuracy", "Validation Accuracy")],
    filename="graphs/dinov2_accuracy.png"
)

# DINOv2 Loss
plot_graph(
    df=df_dino,
    title="DINOv2 Training and Validation Loss",
    y_label="Loss",
    columns=[("Train_Loss", "Training Loss"), ("Val_Loss", "Validation Loss")],
    filename="graphs/dinov2_loss.png"
)

print("Graphs and CSV files generated successfully.")
