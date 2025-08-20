import argparse
import os
import torch
import gradio as gr
import whisper

from omni_speech.model.builder import load_pretrained_model
from omni_speech.conversation import conv_templates
from omni_speech.utils import disable_torch_init
from omni_speech.datasets.preprocess import tokenizer_speech_token


def build_prompt(prompt_text: str) -> str:
    # 與現有流程一致：在文字前加入 <speech>\n，以啟用語音提示的特殊 token 處理
    text = "<speech>\n" + prompt_text
    conv = conv_templates["llama_3"].copy()
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def build_interface(tokenizer, model, device: str, input_type: str, mel_size: int):
    def infer_fn(audio_path: str, prompt_text: str, temperature: float, top_p: float, max_new_tokens: int):
        if not audio_path:
            return "請提供音訊。"

        # 讀取音訊並依 input_type 準備特徵
        speech = whisper.load_audio(audio_path)
        if input_type == "raw":
            speech_tensor = torch.from_numpy(speech)
            if getattr(model.config, "speech_normalize", False):
                speech_tensor = torch.nn.functional.layer_norm(speech_tensor, speech_tensor.shape)
        else:
            # mel 特徵 (T, mel)
            speech_tensor = whisper.pad_or_trim(speech)
            speech_tensor = whisper.log_mel_spectrogram(speech_tensor, n_mels=mel_size).permute(1, 0)

        speech_length = torch.LongTensor([speech_tensor.shape[0]]).unsqueeze(0).to(device)
        speech_tensor = speech_tensor.unsqueeze(0).to(device, dtype=torch.float16)

        # 準備 prompt 與 input_ids
        prompt = build_prompt(prompt_text)
        input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(device)

        # 產生文字（Stage-1 僅文字輸出）
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                speech=speech_tensor,
                speech_lengths=speech_length,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p if temperature > 0 else None,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=128004,
            )

        prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return prediction

    with gr.Blocks(title="LLaMA-Omni Text-only Demo") as demo:
        gr.Markdown("""
        ### LLaMA-Omni（Stage-1）文字輸出 Demo
        - 上傳音檔或使用麥克風錄音
        - 僅輸出文字回應
        """)

        with gr.Row():
            audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Speech Input")
            with gr.Column():
                prompt_in = gr.Textbox(value="Please directly answer the questions in the user's speech", label="Prompt")
                temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Top P")
                max_new_tokens = gr.Slider(1, 1024, value=256, step=16, label="Max New Tokens")

        with gr.Row():
            submit_btn = gr.Button(value="Run", variant="primary")
            clear_btn = gr.Button(value="Clear")

        text_out = gr.Textbox(label="Text Output")

        submit_btn.click(
            infer_fn,
            [audio_in, prompt_in, temperature, top_p, max_new_tokens],
            [text_out]
        )
        clear_btn.click(None, [], [audio_in, text_out], _js="() => [null, '']")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--is-lora", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--input-type", type=str, default="mel", choices=["raw", "mel"])
    parser.add_argument("--mel-size", type=int, default=128)
    args = parser.parse_args()

    disable_torch_init()
    tokenizer, model, _ = load_pretrained_model(
        os.path.expanduser(args.model_path), args.model_base, is_lora=args.is_lora, s2s=False, device=args.device
    )
    demo = build_interface(tokenizer, model, args.device, args.input_type, args.mel_size)
    demo.launch(server_name=args.host, server_port=args.port, share=False)


