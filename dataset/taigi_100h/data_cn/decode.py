import json

# 輸入檔案與輸出檔案名稱
input_file = "/mnt/md0/user_yuze0w0/dataset/taigi_100h/data_cn/answer_ckpt30000_train.json"
output_file = "/mnt/md0/user_yuze0w0/dataset/taigi_100h/data_cn/answer_ckpt30000_trainn.json"

decoded_lines = []

# 逐行讀取並解碼
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():  # 跳過空行
            obj = json.loads(line)  # 自動解碼 \uXXXX
            decoded_lines.append(obj)

# 輸出成新的 JSON 檔（格式化輸出）
with open(output_file, "w", encoding="utf-8") as f:
    for obj in decoded_lines:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"轉換完成！已輸出到 {output_file}")

# python /mnt/md0/user_yuze0w0/dataset/taigi_100h/data_cn/decode.py