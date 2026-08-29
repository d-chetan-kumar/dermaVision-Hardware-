import pandas as pd
import matplotlib.pyplot as plt
import os

# Create directories
os.makedirs('results', exist_ok=True)
os.makedirs('graphs', exist_ok=True)

data = """Epoch,Train_Loss,Train_Accuracy,Val_Loss,Val_Accuracy
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

# Write to CSV
with open('results/dinov2_epoch_history.csv', 'w') as f:
    f.write(data)

# Read into DataFrame
df = pd.read_csv('results/dinov2_epoch_history.csv')

# Plot Accuracy
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Train_Accuracy'], label='Training Accuracy', color='blue', linewidth=2)
plt.plot(df['Epoch'], df['Val_Accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
plt.title('DINOv2 Performance: Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('graphs/dinov2_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot Loss
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Train_Loss'], label='Training Loss', color='red', linewidth=2)
plt.plot(df['Epoch'], df['Val_Loss'], label='Validation Loss', color='green', linewidth=2)
plt.title('DINOv2 Performance: Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('graphs/dinov2_loss.png', dpi=300, bbox_inches='tight')
plt.close()

print("DINOv2 graphs and history successfully generated.")
