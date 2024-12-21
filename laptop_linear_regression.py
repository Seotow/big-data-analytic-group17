import pandas as pd
import numpy as np
import math
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import platform
import psutil
# Bắt đầu đo thời gian
start_time = time.time()

laptops_df = pd.read_csv("laptops_dataset.csv")
print(laptops_df.head(10).to_string())

# Làm sạch dữ liệu
nan_counts = laptops_df.isna().sum()
for column, count in nan_counts.items():
    print(f"{column}: {count}")

selected_columns = [
    "Brand", "Standing screen display size", "Processor", "RAM",
    "Memory Speed", "Hard Drive", "Graphics Coprocessor",
    "Chipset Brand", "Card Description", "Processor Brand",
    "Operating System", "Item Weight", "Price($)"
]

laptops_df_cleaned = laptops_df[selected_columns].dropna()


# Chuyển Memory speed sang Hz
def convert_to_hz(speed):
    speed = "".join(filter(str.isprintable, speed))
    speed = speed.strip()
    if "GHz" in speed:
        return float(speed.replace("GHz", "").strip()) * 1e9
    elif "MHz" in speed:
        return float(speed.replace("MHz", "").strip()) * 1e6
    elif "KHz" in speed:
        return float(speed.replace("KHz", "").strip()) * 1e3
    else:
        return float(speed)


print(laptops_df_cleaned[['Memory Speed']].head(10).to_string(index=True, header=True))
laptops_df_cleaned["Memory Speed"] = laptops_df_cleaned["Memory Speed"].apply(convert_to_hz)
laptops_df_cleaned = laptops_df_cleaned.rename(columns={"Memory Speed": "Memory Speed(Hz)"})
print(laptops_df_cleaned[['Memory Speed(Hz)']].head(10).to_string(index=True, header=True))


# Chuyển trọng lượng sang pounds
def clean_and_convert_weight(weight_str):
    cleaned = re.sub(r"[^\d.]+", "", weight_str).lower()

    if "ounce" in weight_str:
        cleaned = float(cleaned) * 0.0625

    return float(cleaned)


print(laptops_df_cleaned[['Item Weight']].head(10).to_string(index=True, header=True))
laptops_df_cleaned["Item Weight"] = laptops_df_cleaned["Item Weight"].apply(clean_and_convert_weight)
laptops_df_cleaned = laptops_df_cleaned.rename(columns={"Item Weight": "Item Weight(pounds)"})
print(laptops_df_cleaned[['Item Weight(pounds)']].head(10).to_string(index=True, header=True))

# Loại bỏ chuỗi "Inches" trong Standing screen display size
print(laptops_df_cleaned[['Standing screen display size']].head(10).to_string(index=True, header=True))

laptops_df_cleaned["Standing screen display size"] = laptops_df_cleaned["Standing screen display size"].str.replace(
    " Inches", "")
laptops_df_cleaned["Standing screen display size"] = laptops_df_cleaned["Standing screen display size"].str.replace(
    r"[^\d\.]", "", regex=True)
laptops_df_cleaned["Standing screen display size"] = laptops_df_cleaned["Standing screen display size"].astype(
    float).round(2)
laptops_df_cleaned = laptops_df_cleaned.rename(
    columns={"Standing screen display size": "Standing screen display size(Inches)"})
print(laptops_df_cleaned[['Standing screen display size(Inches)']].head(10).to_string(index=True, header=True))


# Làm sạch Ram
def clean_ram(value):
    match = re.search(r"\d+", value)
    if match:
        ram_value = int(match.group())
        if "1 TB" in value:
            return 1024
        elif ram_value % 2 != 0:
            return int(math.pow(2, math.ceil(math.log2(ram_value))))
        else:
            return ram_value


print(laptops_df_cleaned[['RAM']].head(10).to_string(index=True, header=True))
laptops_df_cleaned["RAM"] = laptops_df_cleaned["RAM"].apply(clean_ram)
laptops_df_cleaned.dropna(subset=["RAM"], inplace=True)
laptops_df_cleaned = laptops_df_cleaned.rename(columns={"RAM": "RAM(GB)"})
print(laptops_df_cleaned[['RAM(GB)']].head(10).to_string(index=True, header=True))

# Làm sạch Processor
print(laptops_df_cleaned[['Processor']].head(10).to_string(index=True, header=True))
laptops_df_cleaned["Processor"] = laptops_df_cleaned["Processor"].apply(
    lambda x: re.search(r"(\d+\.\d+)\s*GHz", x).group(1) if re.search(r"(\d+\.\d+)\s*GHz", x) else None)
laptops_df_cleaned.dropna(subset=["Processor"], inplace=True)
laptops_df_cleaned["Processor"] = laptops_df_cleaned["Processor"].astype(float)
laptops_df_cleaned = laptops_df_cleaned.rename(columns={"Processor": "Processor(GHz)"})
print(laptops_df_cleaned[['Processor(GHz)']].head(10).to_string(index=True, header=True))


# Làm sạch Hard Drive
def convert_hard_drive_size(size_str):
    match = re.search(r"(\d+\.?\d*)", size_str)
    if match:
        value = float(match.group(1))
        if ("TB" in size_str) and (not "GB" in size_str):
            value *= 1000
        elif "MB" in size_str:
            value /= 1024
        elif "GB" not in size_str:
            if 10 < value < 500:
                return value
            elif value < 10:
                value *= 1000
            else:
                value /= 1000000000
        return value
    return None


print(laptops_df_cleaned[['Hard Drive']].head(10).to_string(index=True, header=True))
laptops_df_cleaned["Hard Drive"] = laptops_df_cleaned["Hard Drive"].apply(convert_hard_drive_size)
laptops_df_cleaned = laptops_df_cleaned.dropna().reset_index(drop=True)
laptops_df_cleaned = laptops_df_cleaned.rename(columns={"Hard Drive": "Hard Drive(GB)"})
print(laptops_df_cleaned[['Hard Drive(GB)']].head(10).to_string(index=True, header=True))

print("\nDữ liệu sau khi làm sạch: ")
print(laptops_df_cleaned.head(10).to_string())

# Tóm lược dữ liệu
print("\n")
print(laptops_df_cleaned.describe().to_string())

df_numeric = laptops_df_cleaned.select_dtypes(include=['number'])
# Vẽ boxplot cho tất cả các thuộc tính kiểu số
plt.figure(figsize=(15, 10))
for i, col in enumerate(df_numeric, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(y=df_numeric[col], color='skyblue')
    plt.title(f'Mức độ phân tán {col}')
    plt.ylabel(col)

plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()

# Vẽ histogram cho các thuộc tính kiểu số
plt.figure(figsize=(15, 10))
for i, col in enumerate(df_numeric, 1):
    plt.subplot(4, 3, i)
    # Điều chỉnh bins dựa trên đặc điểm của từng cột
    if df_numeric[col].nunique() < 20:
        bins = df_numeric[col].nunique()
    elif df_numeric[col].max() - df_numeric[col].min() > 1000:
        bins = 50
    else:
        bins = 30
    sns.histplot(df_numeric[col], kde=True, bins=bins, color='skyblue')
    plt.title(f'Biểu đồ histogram cho cột {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()

# Chuyển đổi dữ liệu
# Chuyển đổi các cột phân loại sang dạng số bằng Label Encoding
label_encoder = LabelEncoder()

categorical_columns = ["Brand", "Graphics Coprocessor", "Chipset Brand",
                       "Card Description", "Processor Brand", "Operating System"]

# for col in categorical_columns:
#     laptops_df_cleaned[col] = label_encoder.fit_transform(laptops_df_cleaned[col])

mapping_dict = {}
for col in categorical_columns:
    # Lấy danh sách giá trị gốc và giá trị đã mã hóa
    original_values = laptops_df_cleaned[col].unique()
    encoded_values = label_encoder.fit_transform(original_values)
    mapping_dict[col] = dict(zip(original_values, encoded_values))
    laptops_df_cleaned[col] = label_encoder.fit_transform(laptops_df_cleaned[col])

for col, mapping in mapping_dict.items():
    print(f"Mapping dữ liệu của cột '{col}':")
    for original, encoded in list(mapping.items())[:3]:
        print(f"  {original} -> {encoded}")


# Tính hệ số tương quan giữa các thuộc tính và giá trị mục tiêu Price($)
correlation_matrix = laptops_df_cleaned.corr()
price_correlation = correlation_matrix["Price($)"].sort_values(ascending=False)
print("\nTương quan của các thuộc tính với giá (Price($)):")
print(price_correlation)

# Vẽ biểu đồ scatter cho tất cả các thuộc tính với Price($)
plt.figure(figsize=(20, 15))
for i, col in enumerate(laptops_df_cleaned.columns[:-1], 1):
    plt.subplot(5, 3, i)
    sns.scatterplot(x=laptops_df_cleaned[col], y=laptops_df_cleaned["Price($)"], color='skyblue')
    plt.title(f'Mối quan hệ giữa {col} và Price($)')
    plt.xlabel(col)
    plt.ylabel('Price($)')

plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()

# Tạo và huấn luyện mô hình hồi quy
# Tạo các biến đầu vào X và biến mục tiêu y
X = laptops_df_cleaned.drop("Price($)", axis=1)
y = laptops_df_cleaned["Price($)"]


# Handle outliers by clipping the data to the 1st and 99th percentiles
def clip_outliers(df, columns):
    for col in columns:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = np.clip(df[col], lower, upper)
    return df


X = clip_outliers(X, X.columns)


# Create a pipeline to include polynomial features and standard scaling
pipeline = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("linear_regression", LinearRegression())
])


# Chia dữ liệu thành tập huấn luyện và kiểm tra (train/test = 8/2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Khởi tạo và huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)


# Kết thúc đo thời gian
end_time = time.time()
# Tính toán thời gian thực thi
execution_time = end_time - start_time

# Lấy thông tin cơ bản về hệ thống
system_info = {
    "Hệ điều hành": platform.system(),
    "Kiến trúc máy": platform.machine(),
    "Bộ xử lý": platform.processor(),
    "Số lõi CPU": psutil.cpu_count(logical=False),  # Số lõi vật lý
    "Số lõi CPU logic": psutil.cpu_count(logical=True),  # Số lõi logic (bao gồm cả hyperthreading)
    "Bộ nhớ": round(psutil.virtual_memory().total / (1024 ** 3), 2),  # Bộ nhớ tính theo GB
}

# In ra thông số hệ thống
print("\nThông số hệ thống:")
for key, value in system_info.items():
    print(f"{key}: {value}")

# Lấy thông tin về tình trạng sử dụng CPU và bộ nhớ
cpu_usage = psutil.cpu_percent(interval=1)  # Sử dụng CPU theo phần trăm
memory_usage = psutil.virtual_memory().percent  # Sử dụng bộ nhớ theo phần trăm

print(f"\nTình trạng sử dụng CPU: {cpu_usage}%")
print(f"Tình trạng sử dụng bộ nhớ: {memory_usage}%")
print(f"Thời gian thực thi: {execution_time:.4f} giây")