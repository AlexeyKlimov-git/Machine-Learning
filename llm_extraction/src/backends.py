"""Rules — намеренно простой baseline. Qwen работает локально через Transformers."""

import re

from .prompts import messages


class Rules:
    revision = None

    def predict(self, text, examples=()):
        import json

        lower = text.lower()
        products = [
            name
            for name, pattern in (
                ("card", r"карт"),
                ("deposit", r"вклад|депозит"),
                ("transfer", r"перевод"),
            )
            if re.search(pattern, lower)
        ]
        amounts = re.findall(
            r"(?<!\d)(\d+(?:[.,]\d+)?)\s*(?:руб\w*|₽|usd|eur)\b", lower
        )
        category = None
        if re.search(r"не прош|не работа|ошиб|пропал|списали дважды", lower):
            category = "problem"
        elif re.search(r"прошу|пожалуйста|хочу|закрой|открой|заблокир", lower):
            category = "request"
        elif "?" in lower or re.search(r"как |когда |сколько |какая ", lower):
            category = "question"
        return json.dumps(
            {
                "product": products[0] if len(products) == 1 else None,
                "amount": float(amounts[0].replace(",", "."))
                if len(amounts) == 1
                else None,
                "category": category,
            }
        )


class Qwen:
    def __init__(self, model="Qwen/Qwen3-0.6B", revision="main"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(42)
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision
        ).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def predict(self, text, examples=()):
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages(text, examples),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if inputs.input_ids.shape[-1] > 4096:
            raise ValueError("Input too long for this demo budget")
        # Детерминированное декодирование — настройка этого эксперимента,
        # не утверждение об оптимальных параметрах для любых задач Qwen.
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            output[0, inputs.input_ids.shape[-1] :], skip_special_tokens=True
        ).strip()
