import json
import jiwer

def compare_all_cer_methods(file_path):
    """
    從 answer.json 讀取資料，並完整比較三種 jiwer CER 計算方法的輸出。
    """
    predictions = []
    answers = []
    
    print(f"正在從 '{file_path}' 讀取資料...")
    try:
        # 假設 answer.json 是 JSON Lines 格式
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                predictions.append(str(data.get("prediction", "")))
                answers.append(str(data.get("answer", "")))

        if not predictions or not answers:
            print("錯誤：檔案中沒有讀取到任何預測或參考答案。")
            return

        print(f"成功處理 {len(predictions)} 筆資料。")
        print("=" * 60)
        
        # --- 方法二：使用 process_characters (官方推薦的詳細版本) ---
        print("方法二：使用 jiwer.process_characters")
        output_recommended = jiwer.process_characters(answers, predictions)
        print(f"  - CER: {output_recommended.cer:.4f}")
        print(f"  - 詳細資訊: S={output_recommended.substitutions}, D={output_recommended.deletions}, I={output_recommended.insertions}, H={output_recommended.hits}")
        print("-" * 60)

        # --- 方法三：使用 cer (官方推薦的簡潔版本) ---
        print("方法三：使用 jiwer.cer")
        output_simple = jiwer.cer(answers, predictions)
        print(f"  - CER: {output_simple:.4f}")
        print(f"  - 詳細資訊: 無 (僅回傳一個 float 數字)")
        print("=" * 60)
        
        # --- 最終驗證 ---
        print("最終驗證結果：")
        is_consistent = (abs(output_hack.wer - output_recommended.cer) < 1e-9 and
                         abs(output_recommended.cer - output_simple) < 1e-9)
        
        if is_consistent:
             print("🎉 驗證成功！三種方法計算出的最終 CER 數值完全一致。")
        else:
             print("⚠️ 驗證失敗：計算結果不一致，請檢查 jiwer 版本或資料。")

    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{file_path}'")
    except Exception as e:
        print(f"發生未知錯誤: {e}")


if __name__ == "__main__":
    json_file = "/mnt/md0/user_yuze0w0/dataset/taigi_100h/data_cn/answer.json" 
    compare_all_cer_methods(json_file)