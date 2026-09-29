import matplotlib.pyplot as plt

# ==================== DỮ LIỆU ====================
X = [-8.953, -1.855, 8.024, 0.724, 7.042, -3.256, -5.097, -1.274, 1.952, -3.691,
     8.108, -7.156, -9.575, 1.468, 6.757, -3.131, -5.863, 0.541, -3.695, 5.385,
     3.697, 3.333, -0.278, -2.508, 5.084, 6.403, -3.263, -8.536, 1.463, -6.219,
     -9.743, -6.142, -9.746, -2.371, -2.096, -5.412, -3.691, -5.786, -9.576, -7.638,
     -7.614, -9.949, -0.511, -7.295, -7.117, -9.441, 7.375, -3.130, -9.060, 1.462,
     -5.004, -4.562, -4.022, -3.735, 5.443, -6.942, 5.077, -8.222, 1.468, -5.593,
     -6.372, 3.341, -9.961, -4.381, -8.539, -3.559, -6.615, -1.093, -9.105, 9.918,
     2.026, 3.069, 8.094, -1.753, -3.690, 5.390, 5.544, -5.570, -0.954, -2.759,
     -6.399, -0.197, -1.790, 7.510, 6.407, 2.779, 2.116, -7.236, 1.494, -6.214,
     -7.042, -3.834, -5.984, -4.879, -8.453, -1.585, 5.406, 5.074, -6.386, -8.751]

y = [-29.045, -9.048, 18.196, -0.480, 16.176, -10.743, -18.228, -7.218, 2.232, -14.103,
     20.062, -21.689, -31.935, 1.072, 18.519, -11.992, -15.840, 0.074, -13.637, 12.143,
     10.968, 7.609, -4.258, -6.254, 13.515, 15.533, -8.171, -23.977, 1.202, -19.968,
     -29.208, -18.575, -29.125, -8.045, -7.022, -15.910, -14.232, -18.259, -27.710, -20.373,
     -20.863, -28.939, -4.644, -24.033, -24.004, -26.666, 18.986, -9.549, -24.785, 5.612,
     -14.658, -11.776, -15.322, -10.642, 14.244, -18.983, 15.357, -22.904, 4.848, -19.128,
     -17.146, 7.482, -30.739, -11.541, -27.355, -11.733, -18.055, -4.063, -27.231, 27.113,
     3.758, 6.325, 20.087, -8.767, -12.963, 16.466, 16.382, -17.019, -2.911, -10.046,
     -20.785, -1.892, -6.203, 21.222, 14.736, 7.126, 5.580, -22.327, 5.527, -18.882,
     -23.322, -12.160, -18.627, -13.509, -22.991, -6.786, 12.695, 14.630, -16.518, -26.515]

# ==================== 8 HÀM CẦN THIẾT ====================

# 1. Hàm dự đoán y
def predict(X, w, b):
    return [w * x + b for x in X]

# 2. Hàm tính MAE
def calculate_mae(y_true, y_pred):
    N = len(y_true)
    return sum(abs(y_pred[i] - y_true[i]) for i in range(N)) / N

# 3. Hàm tính MSE
def calculate_mse(y_true, y_pred):
    N = len(y_true)
    return sum((y_pred[i] - y_true[i]) ** 2 for i in range(N)) / N

# 4. Hàm tính gradient theo w
def calculate_gradient_w(X, y_true, y_pred, loss_type='mse'):
    N = len(y_true)
    if loss_type == 'mae':
        return sum((1 if y_pred[i] - y_true[i] > 0 else (-1 if y_pred[i] - y_true[i] < 0 else 0)) * X[i] 
                   for i in range(N)) / N
    else:
        return 2 * sum((y_pred[i] - y_true[i]) * X[i] for i in range(N)) / N

# 5. Hàm tính gradient theo b
def calculate_gradient_b(y_true, y_pred, loss_type='mse'):
    N = len(y_true)
    if loss_type == 'mae':
        return sum(1 if y_pred[i] - y_true[i] > 0 else (-1 if y_pred[i] - y_true[i] < 0 else 0) 
                   for i in range(N)) / N
    else:
        return 2 * sum(y_pred[i] - y_true[i] for i in range(N)) / N

# 6. Hàm cập nhật tham số
def update_parameters(w, b, gradient_w, gradient_b, learning_rate):
    return w - learning_rate * gradient_w, b - learning_rate * gradient_b

# 7. Hàm train theo epoch
def train(X, y, learning_rate, epochs, loss_type='mse', w_init=None, b_init=None):
    w = w_init if w_init is not None else 0.496714
    b = b_init if b_init is not None else -0.138264
    
    history = {'epoch': [], 'loss': [], 'w': [], 'b': []}
    
    for epoch in range(epochs):
        y_pred = predict(X, w, b)
        loss = calculate_mae(y, y_pred) if loss_type == 'mae' else calculate_mse(y, y_pred)
        gw = calculate_gradient_w(X, y, y_pred, loss_type)
        gb = calculate_gradient_b(y, y_pred, loss_type)
        w, b = update_parameters(w, b, gw, gb, learning_rate)
        
        history['epoch'].append(epoch + 1)
        history['loss'].append(loss)
        history['w'].append(w)
        history['b'].append(b)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}, w: {w:.6f}, b: {b:.6f}")
    
    return w, b, history

# 8. Hàm lưu lịch sử
def save_history_table(history, output_file, num_rows=5):
    with open(output_file, 'w') as f:
        f.write("Epoch,Loss,w,b,Phase\n")
        total = len(history['epoch'])
        for i in range(num_rows):
            f.write(f"{history['epoch'][i]},{history['loss'][i]:.6f},{history['w'][i]:.6f},{history['b'][i]:.6f},Epoch Đầu\n")
        for i in range(num_rows):
            idx = total//2 - num_rows//2 + i
            f.write(f"{history['epoch'][idx]},{history['loss'][idx]:.6f},{history['w'][idx]:.6f},{history['b'][idx]:.6f},Epoch Giữa\n")
        for i in range(num_rows):
            idx = total - num_rows + i
            f.write(f"{history['epoch'][idx]},{history['loss'][idx]:.6f},{history['w'][idx]:.6f},{history['b'][idx]:.6f},Epoch Cuối\n")

# ==================== QUY TRÌNH THỰC HIỆN ====================

print("=" * 70)
print("HỒI QUY TUYẾN TÍNH VỚI GRADIENT DESCENT")
print("=" * 70)

w_init = 0.496714
b_init = -0.138264
print(f"\nKhởi tạo: w = {w_init:.6f}, b = {b_init:.6f}")

learning_rates = [0.0005, 0.001, 0.005, 0.01]
epochs = 100
results = {}

# 1. Nhập dữ liệu (100 mẫu)
print(f"\n1. Nhập dữ liệu: {len(X)} mẫu")

# 2. Khởi tạo w, b
print(f"2. Khởi tạo w = {w_init}, b = {b_init}")

# 3. Lặp qua các epoch với 4 learning rate
for lr in learning_rates:
    print(f"LEARNING RATE = {lr}")
    w, b, hist = train(X, y, lr, epochs, 'mse', w_init, b_init)
    results[lr] = {'w': w, 'b': b, 'history': hist}
    print(f"Kết quả: w = {w:.6f}, b = {b:.6f}, Loss = {hist['loss'][-1]:.6f}")

# 4. Lưu kết quả
print(f"\n{'=' * 70}")
print("4. LƯU LẠI KẾT QUẢ")
print(f"{'=' * 70}")
save_history_table(results[0.005]['history'], '/mnt/user-data/outputs/summary_mse_lr0005.csv')

# 5. Lặp lại với MAE
print(f"\n{'=' * 70}")
print("5. LẶP LẠI VỚI NHIỀU LEARNING RATE (MAE)")
print(f"{'=' * 70}")

print(f"\nMAE với LR = 0.005:")
w_mae, b_mae, hist_mae = train(X, y, 0.005, epochs, 'mae', w_init, b_init)
print(f"Kết quả: w = {w_mae:.6f}, b = {b_mae:.6f}, Loss = {hist_mae['loss'][-1]:.6f}")

save_history_table(hist_mae, '/mnt/user-data/outputs/summary_mae_lr0005.csv')

# ==================== VẼ HÌH ====================

# Hình 1: Các đường hồi quy
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Linear Regression - Different Learning Rates', fontsize=16, fontweight='bold')

for idx, (lr, ax) in enumerate(zip(learning_rates, axes.flat)):
    w, b = results[lr]['w'], results[lr]['b']
    ax.scatter(X, y, alpha=0.6, s=50, label='Data', color='blue')
    
    x_min, x_max = min(X) - 1, max(X) + 1
    x_range = [x_min + i * (x_max - x_min) / 99 for i in range(100)]
    y_init = [w_init * x + b_init for x in x_range]
    y_final = [w * x + b for x in x_range]
    
    ax.plot(x_range, y_init, 'r--', alpha=0.5, linewidth=2, label='Initial')
    ax.plot(x_range, y_final, 'g-', linewidth=2.5, label='Final')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'LR = {lr}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/01_regression_lines_all_lr.png', dpi=300, bbox_inches='tight')
plt.close()
