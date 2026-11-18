"""Inference adapter. Imported only by an explicit remote evaluation command."""

import copy
import importlib.metadata
import platform
from pathlib import Path

from .benchmark import LABELS
from .download import read_json


def software_versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "transformers", "xlstm", "triton", "accelerate", "huggingface-hub"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


class HFScorer:
    def __init__(self, name: str, spec: dict, artifacts: Path, checkpoint: Path | None = None,
                 max_input_units: int = 8192):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A CUDA GPU with native BF16 support is required")
        self.torch, self.family = torch, spec["family"]
        self.max_input_units = max_input_units
        manifest = read_json(artifacts / "models" / f"{name}.json")
        if manifest["revision"] != spec["revision"] or not manifest["weights"]:
            raise ValueError("Prepare matching full weights first")
        model_path = checkpoint.resolve() if checkpoint else Path(manifest["path"])
        # Adaptation override currently supported only for Qwen full merged checkpoints.
        if checkpoint and self.family != "qwen3":
            raise ValueError("Checkpoint override currently supports Qwen only")
        kwargs = {"trust_remote_code": spec["trust_remote_code"], "local_files_only": True}
        config = AutoConfig.from_pretrained(str(model_path), **kwargs)
        tokenizer_kwargs = dict(kwargs)
        if self.family == "bolmo":
            # Upstream Bolmo opens its original subword tokenizer internally.
            # Redirect to a pinned local dependency rather than a floating Hub branch.
            token_config = copy.deepcopy(config.tokenizer_config)
            dependency_path = manifest["tokenizer_dependency"]["path"]
            if isinstance(token_config, dict):
                token_config["original_identifier"] = dependency_path
            else:
                token_config.original_identifier = dependency_path
            config.tokenizer_config = token_config
            tokenizer_kwargs["tokenizer_config"] = token_config
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), **tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), config=config, torch_dtype=torch.bfloat16, **kwargs).eval().to("cuda")
        self.device = torch.device("cuda")
        self.label_ids = []
        for label in LABELS:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Label {label} is not a single token/byte")
            self.label_ids.append(ids[0])
        self.manifest = manifest
        self.model_path = str(model_path)

    def score(self, prompt: str, n_choices: int = 4) -> tuple[list[float], int]:
        """Prompt only: never append the correct label or candidate answer."""
        torch = self.torch
        if not 2 <= n_choices <= 4:
            raise ValueError("Expected 2..4 choices")
        label_ids = self.label_ids[:n_choices]
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if self.family in ("bolmo", "evabyte"):
            ids = [self.tokenizer.bos_token_id] + ids
        if len(ids) > self.max_input_units:
            raise ValueError(f"Prompt has {len(ids)} input units; truncation is forbidden")
        input_ids = torch.tensor([ids], device=self.device)
        kwargs = {"input_ids": input_ids, "use_cache": False, "return_dict": True}
        if self.family == "evabyte":
            # First next-byte head only. No attention mask for a single unpadded example.
            kwargs["return_all_pred_logits"] = False
        else:
            kwargs["logits_to_keep"] = 1
        with torch.inference_mode():
            logits = self.model(**kwargs).logits[0, -1].float()
            if logits.ndim != 1 or not torch.isfinite(logits).all():
                raise ValueError("Unexpected/non-finite next-unit logits")
            if self.family == "bolmo":
                # 520 = 260 ordinary/special atoms + 260 atoms with a fused boundary.
                # Sum probabilities BEFORE normalization over A/B/C/D.
                half = self.model.config.vocab_size // 2
                if logits.numel() != 2 * half or max(label_ids) >= half:
                    raise ValueError("Unsupported Bolmo output vocabulary")
                selected = torch.stack([torch.logaddexp(logits[i], logits[i + half])
                                        for i in label_ids])
            else:
                selected = logits[label_ids]
            scores = torch.log_softmax(selected, dim=-1).cpu().tolist()
        return scores, len(ids)
