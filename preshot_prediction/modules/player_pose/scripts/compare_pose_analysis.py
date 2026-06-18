import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
camera_name="aps23287107"
# File paths
pre_path = f"pre_pose_analysis/pre_pose_analysis_{camera_name}.csv"
post_path = f"post_pose_analysis/pose_analysis_{camera_name}.csv"

# Load CSVs
df_pre = pd.read_csv(pre_path)
df_post = pd.read_csv(post_path)

# Drop sequence_number for analysis
data_pre = df_pre.drop(columns=["sequence_number"])
data_post = df_post.drop(columns=["sequence_number"])

# Convert to numeric (handle 'nan' strings)
data_pre = data_pre.apply(pd.to_numeric, errors='coerce')
data_post = data_post.apply(pd.to_numeric, errors='coerce')

# Plot distributions for each keypoint
keypoints = data_pre.columns
num_cols = 4
num_rows = int(np.ceil(len(keypoints) / num_cols))
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))
axes = axes.flatten()

for i, kp in enumerate(keypoints):
    total_pre=len(data_pre[kp])
    total_post=len(data_post[kp])
    availability_pre = data_pre[kp].count()  * 100/ total_pre
    availability_post = data_post[kp].count() * 100 / total_post
    print(f"{kp}: Pre-rotation availability: {availability_pre:.2f}%, Post-rotation availability: {availability_post:.2f}%")
    
    
    axes[i].hist(data_pre[kp].dropna()[data_pre[kp]<30], bins=30, alpha=0.5, label='Pre', color='blue')
    axes[i].hist(data_post[kp].dropna()[data_post[kp]<30], bins=30, alpha=0.5, label='Post', color='orange')
    axes[i].set_title(kp)
    axes[i].set_xlabel('Repj Error (pixels)',loc='left')
    axes[i].set_ylabel('Frequency')
    axes[i].legend()
plt.tight_layout()
os.makedirs("pose_analysis", exist_ok=True)
plt.savefig(f"pose_analysis/comparison_histograms_{camera_name}.png")

# --- Availability comparison plot ---
availability_pre_list = [(data_pre[kp].count() * 100 / len(data_pre[kp])) for kp in keypoints]
availability_post_list = [(data_post[kp].count() * 100 / len(data_post[kp])) for kp in keypoints]

plt.figure(figsize=(16, 6))
bar_width = 0.35
x = np.arange(len(keypoints))
plt.plot(x , availability_pre_list, label='Pre', color='blue', alpha=0.7)
plt.plot(x , availability_post_list, label='Post', color='orange', alpha=0.7)
plt.xticks(x, keypoints, rotation=45, ha='right')
plt.ylabel('Availability (%)')
plt.title('Keypoint Availability Comparison (Pre vs Post)')
plt.legend()
plt.tight_layout()
plt.savefig(f"pose_analysis/availability_{camera_name}.png")