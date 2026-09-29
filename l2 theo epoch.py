import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Định nghĩa lớp Linear Regression tu scratch
class LinearRegressionRegularized:
    def __init__(self, lr=0.01, epochs=500, lamda=0.1, penalty=None):
        self.lr = lr
        self.epochs = epochs
        self.lamda = lamda
        self.penalty = penalty
        self.w = None

    def fit(self, X, y, w_init):
        n_samples, n_features = X.shape
        # Khởi tạo trọng số bằng chính giá trị cấu hình ban đầu
        self.w = w_init.copy()
        w_history = [self.w.copy()]
        
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.w)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))

            if self.penalty == 'l1':
                dw += self.lamda * np.sign(self.w)
            elif self.penalty == 'l2':
                dw += self.lamda * self.w

            self.w -= self.lr * dw
            w_history.append(self.w.copy())
            
        return np.array(w_history)

# 2. Tạo dữ liệu giả lập với trọng số gốc thực tế (Giữ nguyên như bài L1)
np.random.seed(42)
X = np.random.randn(200, 5)
w_true = np.array([4.5, -3.0, 1.2, -0.5, 0.1])
y = np.dot(X, w_true) + np.random.randn(200) * 0.2

# 3. Huấn luyện mô hình L2 với Lamda = 0.5
epochs = 500
model_l2 = LinearRegressionRegularized(lr=0.03, epochs=epochs, lamda=0.5, penalty='l2')
# Truyền w_true vào làm trọng số khởi tạo ban đầu
w_history_l2 = model_l2.fit(X, y, w_init=w_true)

# 4. Vẽ đồ thị biến đổi theo Epoch của L2
sns.set_theme(style="whitegrid")
plt.figure(figsize=(11, 6))

features = [
    'Feature 0 (w_init = 4.5)', 
    'Feature 1 (w_init = -3.0)', 
    'Feature 2 (w_init = 1.2)', 
    'Feature 3 (w_init = -0.5)', 
    'Feature 4 (w_init = 0.1)'
]

for i in range(5):
    plt.plot(range(epochs + 1), w_history_l2[:, i], linewidth=2, label=features[i])

plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.title("Sự biến đổi trọng số L2 với Lamda = 0.5", fontsize=12, fontweight='bold')
plt.xlabel("Epoch (Số bước lặp)", fontsize=11)
plt.ylabel("Giá trị Trọng số (w)", fontsize=11)
plt.legend()
plt.show()