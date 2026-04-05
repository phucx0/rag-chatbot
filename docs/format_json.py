import json

# Đọc file đang bị mã hóa \u
with open('train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Ghi lại file với định dạng tiếng Việt nguyên bản
with open('train_vietnamese.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Đã chuyển xong! Giờ bạn có thể mở file train_vietnamese.json để đọc.")