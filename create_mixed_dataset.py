import os
import json

def create_mixed_dataset(output_base_path):
    """
    合併 mandarin_100h 和 taigi_100h 兩個資料集，
    並為它們各自指定不同的 prompt，最後生成混合資料集。
    """
    # --- 資料集設定 ---
    datasets_info = {
        "mandarin": {
            "text_base_path": "/mnt/md0/user_yuze0w0/dataset/mandarin_100h/data",
            "wav_base_path": "/mnt/md0/user_yuze0w0/dataset/mandarin_100h/mandarin_100h_trimmed",
            "prompt": "Provide a word-by-word transcript of the audio recording.",
            "file_suffix": ".wav" # 假設國語的音檔也是 .wav
        },
        "taigi": {
            "text_base_path": "/mnt/md0/user_yuze0w0/dataset/taigi_100h/data_cn",
            "wav_base_path": "/mnt/md0/user_yuze0w0/dataset/taigi_100h/taigi_wav",
            "prompt": "Please translate the user's speech to Chinese.",
            "file_suffix": ".wav"
        }
    }

    # 確保輸出目錄存在
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
        print(f"已建立輸出目錄：{output_base_path}")

    # 遍歷 train, dev, test
    for split in ["train", "dev", "test"]:
        print(f"\n--- 正在處理 '{split}' 資料集 ---")
        
        combined_data = []
        
        # 依序處理每個資料來源 (mandarin, taigi)
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

        # 將合併後的資料寫入新的 JSON 檔案
        output_file_path = os.path.join(output_base_path, f"{split}_mix.json")
        
        print(f"正在將總共 {len(combined_data)} 筆資料寫入: {output_file_path}")
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            json.dump(combined_data, f_out, indent=4, ensure_ascii=False)
        print(f"成功建立 '{output_file_path}'！")

if __name__ == "__main__":
    # --- 設定輸出路徑 ---
    # 您可以指定一個新的路徑，或沿用舊的路徑
    output_directory = "/mnt/md0/user_yuze0w0/dataset/mixed_data"
    
    create_mixed_dataset(output_directory)