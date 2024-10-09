import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest,chi2

# Hàm lọc vị trí
def filter_location(loc):
    result = re.findall("\,\s[A-Z]{2}", loc)
    if len(result):
        return result[0][2:]
    else:
        return loc

# Đọc dữ liệu từ file ODS
data = pd.read_excel("D:\\DHnam4\\XLDL\\job_leveling\\final_project.ods", engine="odf", dtype=str)

# Xử lý dữ liệu
data = data.dropna()  # Xóa hàng có giá trị NaN
data["location"] = data["location"].apply(filter_location)

# Tách nhãn và đặc trưng
target = "career_level"
X = data.drop(target, axis=1)
y = data[target]

# Chia dữ liệu thành tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing dữ liệu
preprocessing = ColumnTransformer(
    transformers=[
        ("title-feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 2)), "title"),
        ("nom_feature", OneHotEncoder(handle_unknown='ignore'), ["location", "function"]),
        ("des_feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95), "description"),
        ("industry_feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 2)), "industry"),
    ])

# Khởi tạo và huấn luyện mô hình
reg = Pipeline(
    steps=[
        ("preprocessing", preprocessing),
        ("feature_select", SelectKBest(chi2, k=800)),
        ("regressor", LogisticRegression())  
    ]
)

# re = reg.fit_transform(X_train)
# print(re.shape())
# Huấn luyện mô hình
reg.fit(X_train, y_train)

# # lưu mô hình
import pickle
pickle.dump(reg, open('jobleveling.pkl', 'wb'))

# # Dự đoán và báo cáo kết quả
# y_predict = reg.predict(X_test)
# print(y_predict)
# report = classification_report(y_test, y_predict)
# print(report)

# Dữ liệu career levels và giá trị tương ứng
# career_levels = [
#     "Senior Specialist / Project Manager", 
#     "Manager / Team Leader", 
#     "Bereichsleiter", 
#     "Director / Business Unit Leader", 
#     "Specialist", 
#     "Managing Director (Small/Medium Company)"
# ]
# values = [4338, 2672, 960, 70, 30, 4]

# # Tạo biểu đồ cột ngang
# plt.figure(figsize=(10, 6))
# plt.barh(career_levels, values, color='skyblue')
# plt.xlabel('Số lượng nhân viên')
# plt.title('Phân phối nhân viên theo cấp bậc sự nghiệp')
# plt.gca().invert_yaxis()  # Đảo ngược trục y để dễ đọc hơn
# plt.tight_layout()

# # Hiển thị biểu đồ
# plt.show()
# Kết quả mô hình sau khi giảm chiều dữ liệu (min_df, max_df, cao hơn so với khi chưa giảm và chạy nhanh hơn)
#                                           precision    recall  f1-score   support

#                         bereichsleiter       0.55      0.45      0.50       192
#          director_business_unit_leader       0.67      0.29      0.40        14
#                    manager_team_leader       0.69      0.65      0.67       534
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.84      0.92      0.88       868
#                             specialist       0.00      0.00      0.00         6

#                               accuracy                           0.77      1615
#                              macro avg       0.46      0.38      0.41      1615
#                           weighted avg       0.75      0.77      0.76      1615

# Kết quả mô hình sau khi giảm chiều dữ liệu và chọn ra những đặc trưng quan trọng ( hiệu suất ko thay đổi nhiều so với bên trên)

#                                             precision    recall  f1-score   support

#                         bereichsleiter       0.53      0.38      0.44       192
#          director_business_unit_leader       0.80      0.29      0.42        14
#                    manager_team_leader       0.68      0.67      0.67       534
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.85      0.92      0.88       868
#                             specialist       0.00      0.00      0.00         6

#                               accuracy                           0.76      1615
#                              macro avg       0.47      0.38      0.40      1615
#                           weighted avg       0.75      0.76      0.75      1615
