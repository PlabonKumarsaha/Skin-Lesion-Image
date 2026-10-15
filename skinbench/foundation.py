"""Open image-text foundation models behind one interface: encode_images(PIL list) and encode_text(str list).

All returned embeddings are L2-normalised; `logit_scale` turns cosine similarity into zero-shot logits.
"""
import sys
from pathlib import Path

import torch

OPEN_CLIP = {  # short name -> open_clip hub id
    "biomedclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "dermlip_vitb16": "hf-hub:redlessone/DermLIP_ViT-B-16",          # gated: needs HF login
}
# DermLIP-PanDerm uses a custom CAE vision encoder that only the authors' open_clip fork can build
# (github.com/SiyuanYan1/Derm1M, cloned into third_party/). Its HF checkpoint holds the full weights.
OPEN_CLIP_FORK = {"dermlip_panderm": "hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256"}
FORK_SRC = Path(__file__).resolve().parents[1] / "third_party/Derm1M/src"
HF_CLIP = {"monet": "suinleelab/monet"}
# Vision-only encoders (linear probe only, no zero-shot): PanDerm-L from the Nature Medicine paper,
# built with the authors' code (github.com/SiyuanYan1/PanDerm, cloned into third_party/).
ROOT = Path(__file__).resolve().parents[1]
PANDERM_SRC = ROOT / "third_party/PanDerm/classification"
PANDERM_L = ROOT / "weights/panderm/panderm_ll_data6_checkpoint-499.pth"


class Foundation:
    def __init__(self, name, device):
        self.name, self.device = name, device
        if name in OPEN_CLIP or name in OPEN_CLIP_FORK:
            if name in OPEN_CLIP_FORK:
                assert "open_clip" not in sys.modules, "load the fork in its own process"
                sys.path.insert(0, str(FORK_SRC))
                import open_clip
                import open_clip.factory as factory
                build = factory.call_PanDerm_base_visual  # skip its separate init file: the HF checkpoint overwrites it
                factory.call_PanDerm_base_visual = lambda path, linear_prob=False: build(None, linear_prob)
                hub = OPEN_CLIP_FORK[name]
            else:
                import open_clip
                hub = OPEN_CLIP[name]
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(hub)
            self.tokenizer = open_clip.get_tokenizer(hub)
            self.logit_scale = float(self.model.logit_scale.detach().exp())
            self.kind = "open_clip"
        elif name in HF_CLIP:
            from transformers import AutoModelForZeroShotImageClassification, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(HF_CLIP[name])
            self.model = AutoModelForZeroShotImageClassification.from_pretrained(HF_CLIP[name])
            self.preprocess = lambda im: self.processor(images=im, return_tensors="pt")["pixel_values"][0]
            self.logit_scale = float(self.model.logit_scale.detach().exp())
            self.kind = "hf"
        elif name == "panderm_l":
            from torchvision import transforms as T
            sys.path.insert(0, str(PANDERM_SRC))
            from models.modeling_finetune import panderm_large_patch16_224
            self.model = panderm_large_patch16_224()
            sd = torch.load(PANDERM_L, map_location="cpu", weights_only=True)
            missing = self.model.load_state_dict({k.replace("encoder.", ""): v for k, v in sd.items()}, strict=False).missing_keys
            assert set(missing) <= {"head.weight", "head.bias"}, missing
            self.preprocess = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                                         T.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225))])  # authors' eval transform
            self.logit_scale = None
            self.kind = "vision"
        else:
            raise ValueError(name)
        self.has_text = self.kind != "vision"
        self.model.to(device).eval()

    @torch.no_grad()
    def encode_pixels(self, pixels):
        x = pixels.to(self.device)
        if self.kind == "vision":
            f = self.model.forward_features(x, is_train=False)
        elif self.kind == "open_clip":
            f = self.model.encode_image(x)
        else:
            f = self.model.get_image_features(pixel_values=x)
        f = f if isinstance(f, torch.Tensor) else f.pooler_output
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu()

    @torch.no_grad()
    def encode_text(self, texts):
        if self.kind == "open_clip":
            f = self.model.encode_text(self.tokenizer(texts).to(self.device))
        else:
            tok = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            f = self.model.get_text_features(**tok)
            f = f if isinstance(f, torch.Tensor) else f.pooler_output
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu()
