import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Định nghĩa lớp Linear Regression tu scratch
class LinearRegressionRegularized:
    def __init__(self, lr=0.01, epochs=1500, lamda=0.1, penalty=None):
        self.lr = lr
        self.epochs = epochs
        self.lamda = lamda
        self.penalty = penalty
        self.w = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Khởi tạo trọng số bằng 0 để đảm bảo tính đồng đều ban đầu
        self.w = np.zeros(n_features)
        
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.w)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))

            if self.penalty == 'l1':
                dw += self.lamda * np.sign(self.w)
            elif self.penalty == 'l2':
                dw += self.lamda * self.w

            self.w -= self.lr * dw

# 2. Tạo dữ liệu giả lập với THIẾT LẬP TRỌNG SỐ MỚI (Tất cả đều khác 0)
np.random.seed(42)
X = np.random.randn(200, 5)

# Định nghĩa các hệ số gốc thực tế (True weights)
w_true = np.array([4.5, -3.0, 1.2, -0.5, 0.1])
# Tạo y dựa trên các hệ số này và cộng thêm chút nhiễu (noise)
y = np.dot(X, w_true) + np.random.randn(200) * 0.2

# 3. Khảo sát các giá trị Lamda (Mở rộng dải Lamda rộng hơn để xem rõ sự triệt tiêu)
lamdas = [0.0, 0.2, 0.5, 1.0, 2.0, 3.5]
w_l1_history = []
w_l2_history = []

for l in lamdas:
    # Huấn luyện L1
    model_l1 = LinearRegressionRegularized(lr=0.03, epochs=1500, lamda=l, penalty='l1')
    model_l1.fit(X, y)
    w_l1_history.append(model_l1.w.copy())
    
    # Huấn luyện L2
    model_l2 = LinearRegressionRegularized(lr=0.03, epochs=1500, lamda=l, penalty='l2')
    model_l2.fit(X, y)
    w_l2_history.append(model_l2.w.copy())

w_l1_history = np.array(w_l1_history)
w_l2_history = np.array(w_l2_history)

# 4. Vẽ đồ thị biểu diễn
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

features = [
    'Feature 0 (w_true = 4.5)', 
    'Feature 1 (w_true = -3.0)', 
    'Feature 2 (w_true = 1.2)', 
    'Feature 3 (w_true = -0.5)', 
    'Feature 4 (w_true = 0.1)'
]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Đồ thị L1 (Lasso)
for i in range(5):
    axes[0].plot(lamdas, w_l1_history[:, i], marker='o', color=colors[i], linewidth=2, label=features[i])
axes[0].axhline(0, color='black', linestyle='--', alpha=0.5)
axes[0].set_title("L1 Regularization (Lasso) - Sap xep va Loc dac trung", fontsize=12, fontweight='bold')
axes[0].set_xlabel(r"He so phat $\lambda$ (Lamda)", fontsize=11)
axes[0].set_ylabel("Gia tri Trong so (w)", fontsize=11)
axes[0].legend()

# Đồ thị L2 (Ridge)
for i in range(5):
    axes[1].plot(lamdas, w_l2_history[:, i], marker='s', color=colors[i], linewidth=2, label=features[i])
axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
axes[1].set_title("L2 Regularization (Ridge) - Nen deu tat ca cac dac trung", fontsize=12, fontweight='bold')
axes[1].set_xlabel(r"He so phat $\lambda$ (Lamda)", fontsize=11)

plt.tight_layout()
plt.show()