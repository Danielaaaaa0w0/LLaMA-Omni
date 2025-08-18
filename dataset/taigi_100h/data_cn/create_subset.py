import json

def create_subset_from_full_json(input_file_path, output_file_path, num_samples):
    """
    從一個標準的 JSON 陣列檔案中讀取資料，取出子集，
    並將其儲存為一個新的、格式正確的 JSON 陣列檔案。
    """
    print(f"正在從標準 JSON 檔案 '{input_file_path}' 讀取資料...")

    try:
        # 一次性讀取整個 JSON 檔案
        with open(input_file_path, 'r', encoding='utf-8') as f_in:
            full_data = json.load(f_in)

        # 確認讀取到的是一個列表 (list)
        if not isinstance(full_data, list):
            print("錯誤：輸入的 JSON 檔案根層級不是一個列表 (陣列)。")
            return
            
        # 從列表中取出前 num_samples 筆資料
        subset_data = full_data[:num_samples]
        
        print(f"成功讀取 {len(full_data)} 筆資料，並已擷取前 {len(subset_data)} 筆。")
        
        print(f"正在將子集資料寫入 '{output_file_path}'...")
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            # 將取出的子集列表寫入新檔案，並進行美化排版
            json.dump(subset_data, f_out, indent=4, ensure_ascii=False)

        print(f"成功建立子集檔案 '{output_file_path}'！")

    except FileNotFoundError:
        print(f"錯誤：找不到輸入檔案 '{input_file_path}'")
    except json.JSONDecodeError as e:
        print(f"錯誤：解析 JSON 時發生錯誤，請檢查檔案是否為標準格式。錯誤訊息: {e}")
    except Exception as e:
        print(f"發生未知錯誤: {e}")


if __name__ == "__main__":
    # --- 設定 ---
    source_file = "/mnt/md0/user_yuze0w0/dataset/taigi_100h/data_cn/train_final.json"
    output_file = "/mnt/md0/user_yuze0w0/dataset/taigi_100h/data_cn/train_100.json"
    samples_to_take = 100

    create_subset_from_full_json(source_file, output_file, samples_to_take)