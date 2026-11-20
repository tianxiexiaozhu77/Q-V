"""Fixed, label-independent prompt ensemble around the released HF model.

The first prompt for every class is copied verbatim from the authors' test
configuration. Four sentence templates are specified before evaluation. The
released model, checkpoint, tokenizer and image processing remain unchanged.
"""

import re

import torch


VERSION = "fixed-five-prompts-logit-mean-v1"
AGGREGATION = "equal_mean_raw_logits_and_raw_maps"
PROMPT_COUNT = 5

# These are countable noun phrases already present in the released prompts.
# Adding an article changes grammar only; it adds no clinical information.
_COUNTABLE_FINDINGS = frozenset({
    "bone fracture",
    "pulmonary mass",
    "lung nodule",
    "hernia",
    "lung opacity",
    "abnormal lesion",
    "calcified granuloma",
    "lung granuloma",
})


def build_prompt_config(spec):
    """Build a JSON-safe protocol using only class names and author prompts.

    No image, label, metric, prevalence, or validation/test result is inspected.
    Reject unfamiliar author sentence structures instead of silently inventing
    a potentially different finding.
    """
    class_names = list(spec["class_names"])
    original_prompts = list(spec["prompts"])
    if not class_names or len(class_names) != len(original_prompts):
        raise ValueError("Expected one author prompt per nonempty class list")
    if any(not isinstance(name, str) or not name.strip() for name in class_names):
        raise ValueError("Class names must be nonempty strings")
    prompts_by_class = []
    for original in original_prompts:
        if not isinstance(original, str):
            raise ValueError("Author prompts must be strings")
        match = re.fullmatch(r"There is\s+(.+?)[.]?", original.strip())
        if match is None:
            raise ValueError(f"Unsupported author prompt structure: {original!r}")
        finding = match.group(1).strip().lower()
        if not finding or any(mark in finding for mark in ".?!;\n"):
            raise ValueError(f"Expected a single finding in author prompt: {original!r}")
        noun_phrase = finding
        if finding in _COUNTABLE_FINDINGS:
            article = "an" if finding[0] in "aeiou" else "a"
            noun_phrase = f"{article} {finding}"
        prompts_by_class.append([
            original,
            f"There is evidence of {noun_phrase}.",
            f"Findings are consistent with {noun_phrase}.",
            f"The chest X-ray shows {noun_phrase}.",
            f"{noun_phrase[0].upper()}{noun_phrase[1:]} is present.",
        ])
    return {
        "version": VERSION,
        "aggregation": AGGREGATION,
        "class_names": class_names,
        "prompts_by_class": prompts_by_class,
        "prompt_count": PROMPT_COUNT,
    }


class PromptEnsembleInference:
    """Cache five text-feature groups and reuse one image forward per batch.

    Logits are averaged before applying a sigmoid. Raw spatial similarity
    scores are averaged independently, before interpolation or sigmoid. Each
    template is passed through the official matching computation separately;
    text embeddings themselves are never averaged.
    """

    @torch.inference_mode()
    def __init__(self, model, tokenizer, config):
        if model.training:
            raise ValueError("Prompt ensembling requires model.eval()")
        if config.get("version") != VERSION or config.get("aggregation") != AGGREGATION:
            raise ValueError("Unsupported prompt-ensemble protocol")
        if config.get("prompt_count") != PROMPT_COUNT:
            raise ValueError(f"This protocol requires exactly {PROMPT_COUNT} prompts")
        self.class_count = len(config["class_names"])
        prompt_rows = config["prompts_by_class"]
        if not self.class_count or len(prompt_rows) != self.class_count:
            raise ValueError("Prompt rows and classes differ")
        if any(len(row) != PROMPT_COUNT for row in prompt_rows):
            raise ValueError("Every class must have exactly five prompts")
        if any(not isinstance(prompt, str) or not prompt.strip()
               for row in prompt_rows for prompt in row):
            raise ValueError("Ensemble prompts must be nonempty strings")
        self.model = model
        self.loss = model.loss_fns["KeyPhraseAlignmentLoss"]
        self.tokens_by_template = []
        self.features_by_template = []
        for template_index in range(PROMPT_COUNT):
            prompts = [row[template_index] for row in prompt_rows]
            tokens = tokenizer(
                prompts, padding=True, truncation=True, return_tensors="pt"
            ).to(model.device)
            split = [
                {key: value[index:index + 1] for key, value in tokens.items()
                 if key in ("input_ids", "attention_mask")}
                for index in range(self.class_count)
            ]
            features, _ = self.loss.compute_text_features(
                split, model.forward_text_model, ddp_gather=False
            )
            if not torch.isfinite(features).all():
                raise RuntimeError("Non-finite cached text features")
            self.tokens_by_template.append(tokens)
            self.features_by_template.append(features)

    def _forward_templates(self, pixels):
        if len(pixels) == 0:
            raise ValueError("An inference batch must contain at least one image")
        vision = self.model.forward_vision_model(pixels)["vision_tokens"]
        if self.loss.layer_norm is not None:
            vision = self.loss.layer_norm(vision)
        if not self.loss.use_vision_cls_token:
            vision = vision[:, 1:]
        template_logits, template_maps = [], []
        for features in self.features_by_template:
            logits, attention = self.loss.compute_t2i_logits(
                features, vision, need_attn_weights=True
            )
            # The official implementation squeezes singleton dimensions.
            # Restore [classes, images] before transposing to [images, classes].
            logits = logits.reshape(self.class_count, len(pixels)).T
            logits = logits / self.loss.loss_temperature.exp()
            scores = torch.stack(attention).mean(dim=0)
            if self.loss.use_vision_cls_token:
                scores = scores[:, :, 1:]
            side = int(scores.shape[-1] ** 0.5)
            if side * side != scores.shape[-1]:
                raise RuntimeError(f"Non-square patch map: {scores.shape}")
            maps = scores.reshape(len(pixels), self.class_count, side, side)
            if not torch.isfinite(logits).all() or not torch.isfinite(maps).all():
                raise RuntimeError("Non-finite template logits or similarity maps")
            template_logits.append(logits)
            template_maps.append(maps)
        return template_logits, template_maps

    @staticmethod
    def _aggregate(template_logits, template_maps):
        logits_by_template = torch.stack(template_logits, dim=-1)
        logits = logits_by_template.mean(dim=-1)
        maps = torch.stack(template_maps, dim=0).mean(dim=0)
        if not torch.isfinite(logits).all() or not torch.isfinite(maps).all():
            raise RuntimeError("Non-finite ensemble logits or similarity maps")
        return {
            "logits": logits,
            "similarity_scores": maps,
            "template_logits": logits_by_template,
        }

    @torch.inference_mode()
    def __call__(self, pixels):
        """Return logits [N,C], maps [N,C,H,W], and template logits [N,C,5]."""
        return self._aggregate(*self._forward_templates(pixels))

    @torch.inference_mode()
    def verify_reference(self, pixels):
        """Check cached inference against each official per-template call.

        Batch sizes one and two exercise the official singleton squeeze cases.
        This verification deliberately repeats the image forward; normal
        __call__ runs the image encoder only once for all five templates.
        """
        if len(pixels) == 0:
            raise ValueError("Reference verification requires at least one image")
        checks = []
        for count in sorted({1, min(2, len(pixels))}):
            subset = pixels[:count]
            template_logits, template_maps = self._forward_templates(subset)
            reference_logits, reference_maps, per_template = [], [], []
            for index, tokens in enumerate(self.tokens_by_template):
                reference = self.model.compute_logits(subset, [tokens])
                expected_logits = reference["logits"].reshape(count, self.class_count)
                expected_maps = reference["similarity_scores"].reshape_as(template_maps[index])
                if not torch.isfinite(expected_logits).all() or not torch.isfinite(expected_maps).all():
                    raise RuntimeError("Non-finite official reference output")
                torch.testing.assert_close(
                    template_logits[index], expected_logits, atol=2e-5, rtol=1e-4
                )
                torch.testing.assert_close(
                    template_maps[index], expected_maps, atol=2e-5, rtol=1e-4
                )
                per_template.append({
                    "template_index": index,
                    "is_author_prompt": index == 0,
                    "max_logit_error": float((template_logits[index] - expected_logits).abs().max()),
                    "max_map_error": float((template_maps[index] - expected_maps).abs().max()),
                })
                reference_logits.append(expected_logits)
                reference_maps.append(expected_maps)
            outputs = self._aggregate(template_logits, template_maps)
            expected_logits = torch.stack(reference_logits, dim=-1).mean(dim=-1)
            expected_maps = torch.stack(reference_maps, dim=0).mean(dim=0)
            torch.testing.assert_close(
                outputs["logits"], expected_logits, atol=2e-5, rtol=1e-4
            )
            torch.testing.assert_close(
                outputs["similarity_scores"], expected_maps, atol=2e-5, rtol=1e-4
            )
            checks.append({
                "batch_size": count,
                "classes": self.class_count,
                "prompt_count": PROMPT_COUNT,
                "templates": per_template,
                "max_logit_error": float((outputs["logits"] - expected_logits).abs().max()),
                "max_map_error": float((outputs["similarity_scores"] - expected_maps).abs().max()),
                "atol": 2e-5,
                "rtol": 1e-4,
            })
        return checks
