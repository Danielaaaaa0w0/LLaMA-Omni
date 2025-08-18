import os
import json
import random

def create_mixed_dataset_shuffled(output_base_path):
    """
    合併 mandarin_100h 和 taigi_100h 兩個資料集，
    為它們各自指定不同的 prompt，並在儲存前將資料隨機打亂，
    最後生成混合資料集。
    """
    # --- 資料集設定 ---
    datasets_info = {
        "mandarin": {
            "text_base_path": "/mnt/md0/user_yuze0w0/dataset/mandarin_100h/data",
            "wav_base_path": "/mnt/md0/user_yuze0w0/dataset/mandarin_100h/mandarin_100h_trimmed",
            "prompt": "Transcribe the following Mandarin speech into Chinese text.",
            "file_suffix": ".wav"
        },
        "taigi": {
            "text_base_path": "/mnt/md0/user_yuze0w0/dataset/taigi_100h/data_cn",
            "wav_base_path": "/mnt/md0/user_yuze0w0/dataset/taigi_100h/taigi_wav",
            "prompt": "Translate the following Taiwanese Hokkien speech into Chinese text.",
            "file_suffix": ".wav"
        }
    }

    # 設定一個固定的隨機種子，確保每次打亂的順序都一樣
    random.seed(42)

    # 確保輸出目錄存在
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
        print(f"已建立輸出目錄：{output_base_path}")

    # 遍歷 train, dev, test
    for split in ["train", "dev", "test"]:
        print(f"\n--- 正在處理 '{split}' 資料集 ---")
        
        combined_data = []
        
        # 依序處理每個資料來源
        for name, info in datasets_info.items():
            text_file_path = os.path.join(info["text_base_path"], split, "text")
            
            if not os.path.exists(text_file_path):
                print(f"警告：找不到檔案 {text_file_path}，跳過 '{name}' 的 '{split}' 部分。")
                continue

            print(f"正在讀取 '{name}' 資料來源從: {text_file_path}")
            count = 0
            human_conversation_value = f"<speech>\n{info['prompt']}"

            with open(text_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(" ", 1)
                    if len(parts) != 2:
                        continue

                    wav_id, text = parts
                    wav_path = os.path.join(info["wav_base_path"], wav_id + info["file_suffix"])

                    combined_data.append({
                        "id": wav_id,
                        "speech": wav_path,
                        "conversations": [
                            {"from": "human", "value": human_conversation_value},
                            {"from": "assistant", "value": text}
                        ]
                    })
                    count += 1
            print(f"成功從 '{name}' 加入 {count} 筆資料。")

        # --- 新增的步驟：隨機打亂合併後的資料 ---
        print(f"正在隨機打亂總共 {len(combined_data)} 筆資料...")
        random.shuffle(combined_data)
        print("資料已成功打亂！")
        # -----------------------------------------

        # 將合併並打亂後的資料寫入新的 JSON 檔案
        output_file_path = os.path.join(output_base_path, f"{split}_mix_shuffled.json")
        
        print(f"正在將資料寫入: {output_file_path}")
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            json.dump(combined_data, f_out, indent=4, ensure_ascii=False)
        print(f"成功建立 '{output_file_path}'！")

if __name__ == "__main__":
    output_directory = "/mnt/md0/user_yuze0w0/dataset/mixed_data"
    create_mixed_dataset_shuffled(output_directory)