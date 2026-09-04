from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / 'fall_dataset' / 'images'
OUTPUT_DIR = ROOT / 'dataset_visualization'
OUTPUT_DIR.mkdir(exist_ok=True)

classes = ['fall', 'not-fall']
counts = {}
images_by_class = {}

for cls in classes:
    cls_dir = DATASET_DIR / cls
    files = sorted(p for p in cls_dir.iterdir() if p.is_file())
    counts[cls] = len(files)
    images_by_class[cls] = files

print('Dataset summary:')
for cls in classes:
    print(f' - {cls}: {counts[cls]} images')
print(f' - total: {sum(counts.values())}')

# Percentage chart
labels = ['Fall', 'Not-fall']
values = [counts['fall'], counts['not-fall']]
total = sum(values)
percentages = [v / total * 100 for v in values]
colors = ['#d62728', '#1f77b4']

# Pie chart for percentages
fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(
    percentages,
    labels=[f'{label} ({p:.1f}%)' for label, p in zip(labels, percentages)],
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.2},
    textprops={'fontsize': 11}
)
ax.set_title('Tỷ lệ % dữ liệu theo lớp', fontsize=14, pad=20)
plt.tight_layout()
pie_path = OUTPUT_DIR / 'dataset_distribution_percent.png'
plt.savefig(pie_path, dpi=200)
plt.close(fig)

# Bar chart with counts and percentages
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(labels, values, color=colors, edgecolor='black', width=0.6)
ax.set_ylabel('Số lượng ảnh')
ax.set_title('Phân bố tập dữ liệu Fall vs Not-fall')
for i, v in enumerate(values):
    ax.text(i, v + 8, f'{v}\n({percentages[i]:.1f}%)', ha='center', va='bottom', fontsize=10)
ax.set_ylim(0, max(values) * 1.25)
plt.tight_layout()
bar_path = OUTPUT_DIR / 'dataset_distribution.png'
plt.savefig(bar_path, dpi=200)
plt.close(fig)

# Sample preview
sample_files = []
for cls in classes:
    files = images_by_class[cls]
    sample_files.extend(files[:3])

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
fig.suptitle('Mẫu ảnh đại diện của từng lớp', fontsize=14)
for ax, img_path in zip(axes.flat, sample_files):
    cls_name = 'fall' if 'fall' in str(img_path.parent) else 'not-fall'
    try:
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f'{cls_name}\n{img_path.name[:24]}', fontsize=8)
    except Exception:
        ax.text(0.5, 0.5, 'Không đọc được ảnh', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(cls_name, fontsize=8)
    ax.axis('off')

for j in range(len(sample_files), len(axes.flat)):
    axes.flat[j].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
preview_path = OUTPUT_DIR / 'dataset_samples.png'
plt.savefig(preview_path, dpi=200)
plt.close(fig)

print(f'\nSaved: {bar_path}')
print(f'Saved: {preview_path}')
