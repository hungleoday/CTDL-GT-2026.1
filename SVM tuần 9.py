import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
df = pd.read_csv('student_performance_dataset')
# 2. Tiền xử lý dữ liệu (Giữ nguyên các cột giống phần Random Forest)
df_model = df.drop(columns=['student_id'])
# Mã hóa các biến dạng chữ sang dạng số bằng LabelEncoder
cat_cols = ['gender', 'parental_education', 'internet_access', 'extracurricular_activities', 'part_time_job']
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = df_model[col].fillna('Unknown')
    df_model[col] = le.fit_transform(df_model[col])
# Tách đặc trưng (X) và nhãn (y)
X = df_model.drop(columns=['final_grade'])
y = df_model['final_grade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 4. Xây dựng Pipeline gồm Chuẩn hóa dữ liệu (StandardScaler) và Mô hình SVM (SVC)
# Việc dùng pipeline giúp tự động scale tập train và tập test tránh bị rò rỉ dữ liệu (data leakage)
svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, random_state=42))
# Huấn luyện mô hình
svm_model.fit(X_train, y_train)
# 5. Dự đoán trên tập Test
y_pred = svm_model.predict(X_test)
# 6. Đánh giá mô hình 
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report (Precision, Recall, F1-Score)")
print(classification_report(y_test, y_pred, zero_division=0))
# 7. Kiểm tra chéo (K-Fold Cross Validation)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svm_model, X, y, cv=kf, scoring='accuracy')
print("\nK-Fold Cross Validation (k=5)")
print(f"Điểm Accuracy từng fold: {np.round(cv_scores, 4)}")
print(f"Accuracy trung bình: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")