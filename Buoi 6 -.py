import numpy as np
import matplotlib.pyplot as plt

# Cấu hình phong cách nền tối cho slide công nghệ
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# Tạo dữ liệu giả lập cho đồ thị lỗi
epochs = np.linspace(1, 100, 100)
# Lỗi Train giảm liên tục
train_loss = 0.5 * np.exp(-epochs/20) + 0.02
# Lỗi Validation giảm rồi tăng ngược lên (Hình chữ U)
val_loss = 0.45 * np.exp(-epochs/15) + 0.00015 * (epochs - 40)**2 + 0.1

# Tìm điểm Sweet Spot (Vị trí Val Loss thấp nhất)
sweet_spot_epoch = np.argmin(val_loss) + 1
min_val_loss = np.min(val_loss)

# Vẽ các đường cong lỗi
ax.plot(epochs, train_loss, label='Train Loss ($E_{train}$)', color='#00adb5', linewidth=2.5)
ax.plot(epochs, val_loss, label='Validation Loss ($E_{CV}$)', color='#ff5722', linewidth=2.5)

# Vạch đường thẳng đứng ngay tại điểm dừng Early Stopping
ax.axvline(x=sweet_spot_epoch, color='#71e175', linestyle='--', linewidth=2, 
           label=f'Early Stopping (Epoch {sweet_spot_epoch})')


# Cấu hình hiển thị trục và nhãn
ax.set_xlabel('Epochs (Vòng lặp)', fontsize=12, labelpad=10)
ax.set_ylabel('Loss (Hàm lỗi)', fontsize=12, labelpad=10)
ax.set_xlim(0, 100)
ax.set_ylim(0, 0.6)

# Làm đẹp khung đồ thị
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.3)

# Hiển thị chú thích (Legend)
ax.legend(loc='upper right', fontsize=11, frameon=True, facecolor='#222831', edgecolor='none')

# Lưu ảnh chất lượng cao
plt.tight_layout()
plt.savefig('early_stopping_slide.png', bbox_inches='tight')
plt.show()