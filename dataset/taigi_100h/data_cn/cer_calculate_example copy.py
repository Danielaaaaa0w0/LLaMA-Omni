import json
import jiwer

def calculate_cer_demo():
    """
    CER 教學版示範
    兩種計算方式：
    1. 全部字元合併計算 (Weighted by total characters)
    2. 逐句計算 CER 再取平均 (Unweighted average of sentence CERs)
    """
    # ===== 固定範例句子 =====
    ref_texts = [
        "我喜歡吃蘋果",
        "今天是個好天氣",
        "明天要去動物園"
    ]
    pred_texts = [
        "我喜愛吃蘋果",    # 替換 "喜歡" → "喜愛"
        "今天是好天氣",    # 刪掉 "個"
        "明天要去動物原"   # 替換 "園" → "原"
    ]

    _calculate_and_print(ref_texts, pred_texts)


def calculate_cer_from_file(file_path):
    """
    從 JSON 檔案讀取資料並計算 CER
    檔案格式需為 JSON Lines，每行包含：
    {
        "answer": "正確答案",
        "prediction": "模型預測"
    }
    """
    ref_texts = []
    pred_texts = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                ref_texts.append(str(data.get("answer", "")))
                pred_texts.append(str(data.get("prediction", "")))

        if not ref_texts or not pred_texts:
            print("❌ 檔案中沒有讀取到任何預測或參考答案。")
            return

        _calculate_and_print(ref_texts, pred_texts)

    except FileNotFoundError:
        print(f"❌ 找不到檔案 '{file_path}'")
    except json.JSONDecodeError:
        print(f"❌ 錯誤：檔案格式不是正確的 JSON Lines 格式")
    except Exception as e:
        print(f"⚠️ 發生未知錯誤: {e}")


def _calculate_and_print(ref_texts, pred_texts):
    """
    核心計算邏輯（共用）
    """
    print(f"✅ 資料共 {len(ref_texts)} 筆")
    print("=" * 60)

    # 顯示範例對照（前3筆）
    print("【範例資料對照】")
    for i, (ref, pred) in enumerate(zip(ref_texts, pred_texts), 1):
        print(f"{i}. 參考答案: {ref}")
        print(f"   模型預測: {pred}")
        if i >= 3:
            break
    print("=" * 60)

    # ===== 方法一：全部字元合併計算 =====
    output_all = jiwer.process_characters(ref_texts, pred_texts)
    S = output_all.substitutions
    D = output_all.deletions
    I = output_all.insertions
    hits = output_all.hits
    N = S + D + hits
    cer_all = (S + D + I) / N

    print("【方式一】全部字元合併計算 CER（加權計算）")
    print(f"總字元數 N: {N}")
    print(f"替換 S: {S}, 刪除 D: {D}, 插入 I: {I}")
    print(f"手算公式: CER = ({S} + {D} + {I}) / {N} = {cer_all:.4f}")
    print(f"jiwer.process_characters(ref_texts, pred_texts) = {output_all.cer:.4f}")

    print(f"或是可以直接用 jiwer.cer(ref_texts, pred_texts) 來計算 CER:")
    print(f"jiwer.cer(ref_texts, pred_texts) = {jiwer.cer(ref_texts, pred_texts):.4f}")
    print("=" * 60)

    # ===== 方法二：逐句計算 CER 再取平均 =====
    cer_per_sentence = []
    for ref, pred in zip(ref_texts, pred_texts):
        o = jiwer.process_characters([ref], [pred])
        N_i = o.substitutions + o.deletions + o.hits
        cer_i = (o.substitutions + o.deletions + o.insertions) / N_i if N_i > 0 else 0
        cer_per_sentence.append(cer_i)

    cer_avg_sentence = sum(cer_per_sentence) / len(cer_per_sentence)

    print("【方式二】逐句計算 CER 再取平均（不加權）")
    for idx, cer_val in enumerate(cer_per_sentence[:3], 1):
        print(f"句子 {idx} CER: {cer_val:.4f}")
    if len(cer_per_sentence) > 3:
        print("... (其餘略)")
    print(f"平均 CER: {cer_avg_sentence:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    # 範例：直接跑教學版
    # calculate_cer_demo()

    # 範例：用使用者 JSON 檔案計算
    json_file = "/path/to/your/answer.json"
    calculate_cer_from_file(json_file)
