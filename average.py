import pandas as pd

filepath = "C:/Users/trung/OneDrive - Hanoi University of Science and Technology/Documents/Hoc Tap/Embedded/E-Nose/New_Data/Dutch Laydy/Offset/Lan 4/06202007.CSV"
file_savepath = "C:/Users/trung/OneDrive - Hanoi University of Science and Technology/Documents/Hoc Tap/Embedded/E-Nose/New_Data/Dutch Laydy/Offset/Lan 4/dutchlady_25_average_4.CSV"
# Đọc dữ liệu từ file CSV
data = pd.read_csv(filepath)

# Tính toán trung bình của từng cửa sổ 4 mẫu và lưu vào mảng mới
sliding_avg = [data[i:i+4].mean() for i in range(100, 200, 4)]

# Chuyển kết quả thành DataFrame
result = pd.DataFrame(sliding_avg)

# Lưu kết quả vào file CSV mới
result.to_csv(file_savepath, index=False)
