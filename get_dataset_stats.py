import os
import wave
from contextlib import closing

def get_dataset_stats(dataset_base_path):
    """
    計算 train, dev, test 資料集的音檔數量與總時長。
    """
    text_base_path = os.path.join(dataset_base_path, "data_cn")
    wav_base_path = os.path.join(dataset_base_path, "taigi_wav")

    print("開始計算資料集統計資訊...")
    print("-" * 30)

    for split in ["dev"]:
        text_file_path = os.path.join(text_base_path, split, "text")

        if not os.path.exists(text_file_path):
            print(f"警告：找不到檔案 {text_file_path}，跳過 '{split}' 資料集。")
            continue

        total_files = 0
        total_duration_seconds = 0.0

        print(f"正在處理 '{split}' 資料集...")
        with open(text_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue

                wav_id = parts[0]
                wav_path = os.path.join(wav_base_path, f"{wav_id}.wav")

                if os.path.exists(wav_path):
                    total_files += 1
                    try:
                        with closing(wave.open(wav_path, 'r')) as audio_file:
                            frames = audio_file.getnframes()
                            rate = audio_file.getframerate()
                            duration = frames / float(rate)
                            total_duration_seconds += duration
                    except wave.Error as e:
                        print(f"錯誤：無法讀取音檔 {wav_path}: {e}")
                else:
                    print(f"警告：找不到對應的音檔 {wav_path}")

        # 將總秒數轉換為 小時:分鐘:秒 的格式
        hours = int(total_duration_seconds // 3600)
        minutes = int((total_duration_seconds % 3600) // 60)
        seconds = total_duration_seconds % 60
        
        print(f"'{split}' 資料集統計結果：")
        print(f"  - 音檔總數：{total_files} 個")
        print(f"  - 音檔總時長：{hours} 小時 {minutes} 分鐘 {seconds:.2f} 秒")
        print("-" * 30)

if __name__ == "__main__":
    # --- 請確認您的資料集根目錄路徑是否正確 ---
    dataset_base_path = "/mnt/md0/user_yuze0w0/dataset/taigi_100h"
    
    get_dataset_stats(dataset_base_path)