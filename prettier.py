import json
input_file = '/mnt/md0/user_yuze0w0/LLAMA_tw-zh/LLaMA-Omni/mandarin_100h_json/answer_ckpt139270_dev.json'
output_file = '/mnt/md0/user_yuze0w0/LLAMA_tw-zh/LLaMA-Omni/mandarin_100h_json/answer_ckpt139270_dev.txt'
with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line)
        fout.write(f"{data['question_id']}\n")
        fout.write("預測：")
        fout.write(f"{data['prediction']}\n")
        fout.write("正解：")
        fout.write(f"{data['answer']}\n\n")
print("轉換完成！")

# python /mnt/md0/user_yuze0w0/dataset/taigi_100h/data_cn/prettier.py