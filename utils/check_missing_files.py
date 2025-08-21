import os

def check_missing_files(dataset_base_path):
    """
    檢查 dev 和 test 資料集的 text 檔案中所記錄的音檔是否存在。
    """
    text_base_path = os.path.join(dataset_base_path, "data_cn")
    wav_base_path = os.path.join(dataset_base_path, "taigi_wav")

    print("開始排查 dev 與 test 資料集的遺失音檔...")
    print("=" * 40)

    for split in ["dev", "test"]:
        text_file_path = os.path.join(text_base_path, split, "text")
        
        if not os.path.exists(text_file_path):
            print(f"警告：找不到 {text_file_path}，無法檢查 '{split}' 資料集。")
            continue

        print(f"正在檢查 '{split}' 資料集...")
        missing_count = 0
        total_count = 0

        with open(text_file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                total_count += 1
                parts = line.split(" ", 1)
                wav_id = parts[0]
                expected_wav_path = os.path.join(wav_base_path, f"{wav_id}.wav")

                if not os.path.exists(expected_wav_path):
                    if missing_count == 0:
                        # 只在第一次找到遺失檔案時印出標頭
                        print(f"在 '{split}' 資料集中找到以下遺失的音檔：")
                    print(f"  - [行號 {i+1}] 找不到: {expected_wav_path}")
                    missing_count += 1
        
        if missing_count == 0:
            print(f"太好了！'{split}' 資料集中的 {total_count} 個音檔全部都存在。")
        else:
            print(f"'{split}' 資料集檢查完畢，共找到 {missing_count} / {total_count} 個遺失音檔。")
        
        print("-" * 40)

if __name__ == "__main__":
    # --- 請確認您的資料集根目錄路徑是否正確 ---
    dataset_base_path = "/mnt/md0/user_yuze0w0/dataset/taigi_100h"
    
    check_missing_files(dataset_base_path)