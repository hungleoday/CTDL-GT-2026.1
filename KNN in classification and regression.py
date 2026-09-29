import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# Mỗi hàng là một sinh viên: [Giờ học trực tuyến, Giờ tự học]
X_train = np.array([
    [2, 3], 
    [3, 2],
    [5, 6], 
    [6, 5],  
    [8, 9], 
    [9, 8]   
])
# Nhãn cho bài toán Classification (0: Trượt, 1: Đỗ)
y_class = np.array([0, 0, 1, 1, 1, 1])
# Nhãn cho bài toán Regression (Điểm số thi thực tế từ 0 đến 10)
y_reg = np.array([3.5, 4.0, 6.5, 7.0, 9.0, 9.5])
# Sinh viên mới cần dự đoán
X_new = np.array([[7, 5]])
# Chọn K = 3 cho cả 2 bài toán
K = 3
# 2. KNN trong Classification (Phân loại Đỗ/Trượt)
# Khởi tạo mô hình KNN Classification với K=3, sử dụng khoảng cách Euclid
knn_clf = KNeighborsClassifier(n_neighbors=K, metric='euclidean')
knn_clf.fit(X_train, y_class)
# Dự đoán nhãn
pred_class = knn_clf.predict(X_new)
# Xem 3 điểm gần nhất là những điểm nào
distances_clf, indices_clf = knn_clf.kneighbors(X_new)
print("[CLASSIFICATION]")
print(f"Các chỉ số của {K} điểm gần nhất trong tập train: {indices_clf[0]}")
print(f"Nhãn của {K} điểm gần nhất đó: {y_class[indices_clf[0]]}")
print(f"Kết quả dự đoán: { 'Đỗ (1)' if pred_class[0] == 1 else 'Trượt (0)' }")
# 3. KNN REGRESSION (Dự đoán điểm số cụ thể)
# Khởi tạo mô hình KNN Regressor với K=3, mặc định tính trung bình cộng (weights='uniform')
knn_reg = KNeighborsRegressor(n_neighbors=K, metric='euclidean')
knn_reg.fit(X_train, y_reg)
pred_reg = knn_reg.predict(X_new)
distances_reg, indices_reg = knn_reg.kneighbors(X_new)
print("[REGRESSION]")
print(f"Các chỉ số của {K} điểm gần nhất trong tập train: {indices_reg[0]}")
print(f"Điểm số của {K} điểm gần nhất đó: {y_reg[indices_reg[0]]}")
print(f"Điểm số dự đoán: {pred_reg[0]:.2f}")