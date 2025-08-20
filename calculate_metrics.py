import json
import sacrebleu
import jiwer

def calculate_metrics_final(file_path):
    """
    使用 jiwer 官方推薦的最佳實踐來計算 BLEU 和 CER 分數。
    - 直接將句子列表傳入函式庫，不手動拼接。
    - 使用 jiwer.process_characters 處理中文，語意最清晰。
    """
    predictions = []
    answers = []
    
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
        print("=" * 40)

        # --- 1. 計算 BLEU 分數 (維持原樣) ---
        references_for_bleu = [answers]
        bleu = sacrebleu.corpus_bleu(predictions, references_for_bleu, tokenize='zh')
        print("BLEU 分數評估結果:")
        print(f"  - BLEU Score: {bleu.score:.2f}")
        print("-" * 40)

        # --- 2. 計算 CER (使用最佳實踐) ---
        # 直接將句子列表傳入 process_characters
        # jiwer 會自動將每個句子視為字元序列來進行比較和計算
        output = jiwer.process_characters(answers, predictions)
        
        print("使用process_characters計算CER (字元錯誤率) 評估結果:")
        print(f"  - CER: {output.cer:.4f}  (越低越好)")
        print(f"  - 總字元數 (參考): {output.substitutions + output.deletions + output.hits}")
        print(f"  - 替換 (Substitutions): {output.substitutions}")
        print(f"  - 刪除 (Deletions): {output.deletions}")
        print(f"  - 插入 (Insertions): {output.insertions}")
        print("=" * 60)
        
        print("使用jiwer.cer計算CER (字元錯誤率) 評估結果:")
        output_simple = jiwer.cer(answers, predictions)
        print(f"  - CER: {output_simple:.4f}")
        print(f"  - 詳細資訊: 無 (僅回傳一個 float 數字)")
        print("=" * 60)

    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{file_path}'")
    except Exception as e:
        print(f"發生未知錯誤: {e}")

if __name__ == "__main__":
    json_file = "/mnt/md0/user_yuze0w0/LLAMA_tw-zh/LLaMA-Omni/mandarin_100h_json/answer_ckpt139270_dev.json" 
    calculate_metrics_final(json_file)

# python /mnt/md0/user_yuze0w0/dataset/taigi_100h/data_cn/calculate_metrics.py
# python /mnt/md0/user_yuze0w0/LLAMA_tw-zh/LLaMA-Omni/calculate_metrics.py
