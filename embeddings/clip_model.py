import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from utils.config import CLIP_MODEL


class CLIPEmbedder:
    """
    CLIP ViT-B/32 wrapper.
    Produces 512-dimensional L2-normalised embeddings for images and text.
    Because vectors are L2-normalised, cosine similarity == dot product.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP ({CLIP_MODEL}) on {self.device}...")
        self.model     = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.model.eval()
        print("CLIP ready.")

    def _to_tensor(self, feat) -> torch.Tensor:
        """
        Normalise output from CLIP model calls.
        Older transformers versions return a BaseModelOutputWithPooling object
        instead of a raw tensor — this handles both cases.
        """
        if isinstance(feat, torch.Tensor):
            return feat
        if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
            return feat.pooler_output
        if hasattr(feat, "last_hidden_state"):
            return feat.last_hidden_state[:, 0, :]
        raise ValueError(f"Cannot extract tensor from CLIP output: {type(feat)}")

    def embed_image(self, image_path: str) -> list | None:
        """Generate 512-dim embedding from an image file. Returns list or None on error."""
        try:
            image  = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                raw  = self.model.get_image_features(**inputs)
                feat = self._to_tensor(raw)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.squeeze().cpu().numpy().tolist()
        except Exception as e:
            print(f"  Embed error [{image_path}]: {e}")
            return None

    def embed_text(self, text: str) -> np.ndarray:
        """Generate 512-dim embedding from a text string. Returns numpy array."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            raw  = self.model.get_text_features(**inputs)
            feat = self._to_tensor(raw)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze().cpu().numpy()