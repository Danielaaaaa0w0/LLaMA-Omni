import os
import json

def create_data_json_new_format(dataset_base_path, output_base_path, prompt):
    """
    這個函式會讀取 text 檔案，並依照指定的 conversation 格式
    產生 train, dev, test 的 JSON 檔案。
    """
    wav_base_path = os.path.join(dataset_base_path, "taigi_wav")
    text_base_path = os.path.join(dataset_base_path, "data_cn")

    # 確保輸出目錄存在
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    # 根據使用者指定的 prompt 組合 human 的對話內容
    human_conversation_value = f"<speech>\n{prompt}"

    for split in ["train", "dev", "test"]:
        text_file_path = os.path.join(text_base_path, split, "text")
        output_json_path = os.path.join(output_base_path, f"{split}_final.json") # 檔名加上 _final 以區別

        if not os.path.exists(text_file_path):
            print(f"警告：找不到檔案 {text_file_path}，跳過此部分。")
            continue

        data = []
        print(f"正在處理 {text_file_path}...")
        with open(text_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    print(f"警告：格式錯誤，跳過此行：{line}")
                    continue
                
                wav_id, text = parts
                wav_path = os.path.join(wav_base_path, f"{wav_id}.wav")

                # 依照新的格式組合 JSON 物件
                data.append({
                    "id": wav_id,
                    "speech": wav_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": human_conversation_value
                        },
                        {
                            "from": "assistant",
                            "value": text
                        }
                    ]
                })
        
        # 將整理好的資料寫入 JSON 檔案
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"成功建立 {output_json_path}，共包含 {len(data)} 筆資料。")

if __name__ == "__main__":
    # --- 請確認以下路徑是否正確 ---
    # 資料集根目錄
    dataset_base_path = "/mnt/md0/user_yuze0w0/dataset/taigi_100h"
    # 輸出 JSON 檔案的目錄
    output_base_path = "/mnt/md0/user_yuze0w0/dataset/taigi_100h/data_cn"
    # 您指定的 prompt
    prompt = "Please translate the user's speech to Chinese."
    
    create_data_json_new_format(dataset_base_path, output_base_path, prompt)