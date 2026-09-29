import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv(r'D:\HUST\Python-Project\student_performance_dataset.csv')

# Kiểm tra nhanh dữ liệu đầu vào
print(f"Kích thước tập dữ liệu: {df.shape}")

# 2. Tiền xử lý dữ liệu (Preprocessing)
# Loại bỏ cột 'student_id' (không mang ý nghĩa dự đoán) và 'final_exam_score' (cột điểm số chi tiết gây lộ nhãn final_grade)
df_model = df.drop(columns=['student_id', 'final_exam_score'])

# Xử lý giá trị khuyết thiếu ở cột 'parental_education' (nếu có) bằng giá trị xuất hiện nhiều nhất (mode)
if df_model['parental_education'].isnull().sum() > 0:
    df_model['parental_education'] = df_model['parental_education'].fillna(df_model['parental_education'].mode()[0])

# Mã hóa các biến dạng chữ (Categorical variables) sang dạng số bằng LabelEncoder
cat_cols = ['gender', 'parental_education', 'internet_access', 'extracurricular_activities', 'part_time_job']
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])

# Tách đặc trưng (Features - X) và biến mục tiêu (Target - y)
X = df_model.drop(columns=['final_grade'])
y = df_model['final_grade']

# 3. Chia tập dữ liệu thành tập huấn luyện (Train) và tập kiểm tra (Test) theo tỷ lệ 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Khởi tạo và huấn luyện mô hình Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Dự đoán trên tập Test
y_pred = rf_model.predict(X_test)

# 6. Đánh giá mô hình theo tiêu chuẩn Tuần 7/8
print("\n================ ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST ================")
print(f"Accuracy (Độ chính xác tổng thể): {accuracy_score(y_test, y_pred):.4f}")

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report (Precision, Recall, F1-Score) ---")
print(classification_report(y_test, y_pred))

# 7. Kiểm tra chéo (K-Fold Cross Validation)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='accuracy')
print("\n--- K-Fold Cross Validation (k=5) ---")
print(f"Điểm Accuracy từng fold: {np.round(cv_scores, 4)}")
print(f"Accuracy trung bình: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 8. Xem mức độ quan trọng của các đặc trưng (Feature Importance)
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- Độ quan trọng của các đặc trưng (Feature Importance) ---")
print(feature_importances)