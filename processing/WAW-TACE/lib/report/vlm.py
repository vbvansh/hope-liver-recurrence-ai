"""VLM backend — the frozen reasoner.

Three backends:
  dryrun   : no model; returns a deterministic stub JSON so the whole pipeline
             (memory + context assembly + write-back) is runnable/verifiable.
  hf       : local transformers (image-text-to-text), e.g. Qwen2.5-VL or MedGemma.
  endpoint : OpenAI-compatible chat endpoint (e.g. vLLM serving the model).

Same class works for the S2 image-read model and the S3 fusion reasoner; the
pipeline picks which checkpoint to load per pass.
"""
from __future__ import annotations
import json
import re
from typing import List, Optional, Tuple


class VLMClient:
    def __init__(self, backend: str = "dryrun",
                 model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 endpoint: Optional[str] = None, max_new_tokens: int = 1200):
        self.backend = backend
        self.model = model
        self.endpoint = endpoint
        self.max_new_tokens = max_new_tokens
        self._pipe = None

    def generate(self, system: str, user: str, images: Optional[List] = None) -> str:
        if self.backend == "dryrun":
            return self._dryrun(user)
        if self.backend == "hf":
            return self._hf(system, user, images or [])
        if self.backend == "endpoint":
            return self._endpoint(system, user, images or [])
        raise ValueError(f"unknown backend {self.backend}")

    @staticmethod
    def _img_data_url(im) -> str:
        import base64
        import io
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _hf(self, system, user, images):
        if self._pipe is None:
            from transformers import pipeline
            import torch
            self._pipe = pipeline("image-text-to-text", model=self.model,
                                  torch_dtype=torch.bfloat16, device_map="auto")
        import torch
        torch.manual_seed(42)            # MR-RATE-style reproducibility (temp 0 / seed)
        content = [{"type": "image", "image": im} for (_lbl, im) in images]
        content.append({"type": "text", "text": user})
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": content}]
        out = self._pipe(text=messages, max_new_tokens=self.max_new_tokens,
                         do_sample=False)
        return out[0]["generated_text"][-1]["content"]

    def _endpoint(self, system, user, images):
        import urllib.request
        if images:
            content = [{"type": "image_url",
                        "image_url": {"url": self._img_data_url(im)}}
                       for (_lbl, im) in images]
            content.append({"type": "text", "text": user})
            user_msg = {"role": "user", "content": content}
        else:
            user_msg = {"role": "user", "content": user}
        body = json.dumps({"model": self.model, "temperature": 0.0,
                           "max_tokens": self.max_new_tokens,
                           "messages": [{"role": "system", "content": system},
                                        user_msg]}).encode()
        req = urllib.request.Request(self.endpoint, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def _dryrun(self, user):
        if "Verify and output the QC JSON" in user:
            return json.dumps({"passed": True, "issues": [],
                               "suggested_label": "unchanged"})
        if "Output the image-read JSON" in user:   # S2 stub
            return ("<think>DRY-RUN S2 stub.</think>\n" + json.dumps({
                "image_burden_direction": "stable", "image_confidence": 0.0,
                "labels": {}, "findings": {
                    "target_tumor": "STUB", "new_lesions": "STUB",
                    "lymphadenopathy": "STUB", "metastasis": "STUB",
                    "necrosis_hemorrhage": "STUB"},
                "uncertainty": "DRY-RUN: no image read performed."}))
        return ("<think>DRY-RUN stub: no model called. Context assembled OK.</think>\n"
                + json.dumps({
                    "visit_id": "STUB", "phase": "STUB", "timeline_position": "STUB",
                    "comparison": {"baseline": "STUB", "nadir": "STUB", "prior": "STUB"},
                    "measurements": {}, "findings": {
                        "target_tumor": "STUB", "new_lesions": "STUB",
                        "lymphadenopathy": "STUB", "metastasis": "STUB",
                        "necrosis_hemorrhage": "STUB"},
                    "confounders": {"treatment_effect_risk": "low",
                                    "rationale": "STUB"},
                    "progression_label": "SD", "confidence": 0.0,
                    "impression": "DRY-RUN: pipeline wiring verified; no model output.",
                    "carry_forward": {"updated_nadir": {"id": "STUB", "value": None},
                                      "watch_items": ["replace dryrun with hf/endpoint"],
                                      "pending_confirmation": False}}, indent=2))


def parse_report(text: str) -> Tuple[dict, str]:
    """Strip <think>..</think>, extract the JSON object. Returns (report, think)."""
    think = ""
    mt = re.search(r"<think>(.*?)</think>", text, re.S)
    if mt:
        think = mt.group(1).strip()
        text = text[mt.end():]
    ms = re.search(r"\{.*\}", text, re.S)
    if not ms:
        return {"_parse_error": True, "raw": text[:2000]}, think
    try:
        return json.loads(ms.group(0)), think
    except Exception as e:
        return {"_parse_error": str(e), "raw": ms.group(0)[:2000]}, think
