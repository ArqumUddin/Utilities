"""
DINOtxt vision-language model wrapper.

Supports zero-shot image classification and segmentation using DINOv3.
DINOtxt enables text-guided visual understanding at both image and patch levels.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union
from PIL import Image

from ..data.structures import Classification, SegmentationMask
from ..data.padding import ImagePaddingInfo
from .base import pad_image_to_square

from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from dinov3.data.transforms import make_classification_eval_transform

class DINOtxtModel:
    """
    Wrapper for DINOtxt vision-language model.

    Supports:
    - Zero-shot image classification
    - Patch-level segmentation with text prompts
    - Dynamic text prompts per inference
    """
    def __init__(
        self,
        model_name: str = "dinov3_vitl16_dinotxt",
        confidence_threshold: float = 0.3,
        device: Optional[str] = None,
        text_prompts: Optional[List[str]] = None
    ):
        """
        Initialize DINOtxt model.

        Args:
            model_name: Model identifier (currently only supports dinov3_vitl16_dinotxt)
            confidence_threshold: Minimum confidence score for predictions
            device: Device for inference ('cuda' or 'cpu')
            text_prompts: Default text prompts for classification/segmentation
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.text_prompts = text_prompts or ["object", "background"]

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading DINOtxt model: {model_name} on {self.device}")

        try:
            self.model, self.tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
            self.model = self.model.to(self.device)
            self.model.eval()

            self.image_transform = make_classification_eval_transform()

            print(f"DINOtxt model loaded successfully: {model_name}")

        except Exception as e:
            raise RuntimeError(f"Failed to load DINOtxt model {model_name}: {e}")

    @staticmethod
    def pad_image_to_square(image: np.ndarray) -> Tuple[np.ndarray, ImagePaddingInfo]:
        """Pad image to square - delegates to base function."""
        return pad_image_to_square(image)

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for DINOtxt inference.

        Args:
            image: RGB image as numpy array

        Returns:
            Preprocessed image tensor
        """
        pil_image = Image.fromarray(image)
        image_tensor = self.image_transform(pil_image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        return image_tensor

    def _encode_texts(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Encode text prompts using DINOtxt tokenizer.

        Args:
            text_prompts: List of text descriptions

        Returns:
            Encoded text features
        """
        tokenized_texts = self.tokenizer.tokenize(text_prompts).to(self.device)
        with torch.autocast(self.device, dtype=torch.float16 if self.device == 'cuda' else torch.float32):
            with torch.no_grad():
                text_features = self.model.encode_text(tokenized_texts)

        return text_features

    def predict(
        self,
        image: np.ndarray,
        text_prompts: Optional[List[str]] = None,
        top_k: int = 5,
        return_padding_info: bool = False
    ) -> Union[List[Classification], Tuple[List[Classification], ImagePaddingInfo]]:
        """
        Perform zero-shot image classification.

        Args:
            image: RGB image as numpy array
            text_prompts: Text descriptions of classes (required)
            top_k: Return top-k predictions
            return_padding_info: If True, return (classifications, padding_info) tuple

        Returns:
            List of Classification objects, sorted by confidence (descending)
        """
        if text_prompts is None:
            text_prompts = self.text_prompts

        if not text_prompts:
            raise ValueError("text_prompts must be provided for DINOtxt classification")

        padded_image, padding_info = self.pad_image_to_square(image)
        image_tensor = self._preprocess_image(padded_image)
        text_features = self._encode_texts(text_prompts)

        with torch.autocast(self.device, dtype=torch.float16 if self.device == 'cuda' else torch.float32):
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)

        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        similarity = (text_features @ image_features.T).squeeze()
        probabilities = F.softmax(similarity, dim=0).cpu().numpy()

        classifications = []
        for idx, (class_name, confidence) in enumerate(zip(text_prompts, probabilities)):
            if confidence >= self.confidence_threshold:
                classification = Classification(
                    class_name=class_name,
                    confidence=float(confidence),
                    class_id=idx
                )
                classifications.append(classification)

        classifications.sort(key=lambda x: x.confidence, reverse=True)
        classifications = classifications[:top_k]

        if return_padding_info:
            return classifications, padding_info
        return classifications

    def predict_segmentation(
        self,
        image: np.ndarray,
        text_prompts: Optional[List[str]] = None,
        return_padding_info: bool = False,
        output_size: Optional[Tuple[int, int]] = None
    ) -> Union[List[SegmentationMask], Tuple[List[SegmentationMask], ImagePaddingInfo]]:
        """
        Perform patch-level segmentation with text prompts.

        Args:
            image: RGB image as numpy array
            text_prompts: Text descriptions of classes (required)
            return_padding_info: If True, return (masks, padding_info) tuple
            output_size: Target size for output masks (H, W). If None, uses image size

        Returns:
            List of SegmentationMask objects, one per text prompt
        """
        if text_prompts is None:
            text_prompts = self.text_prompts

        if not text_prompts:
            raise ValueError("text_prompts must be provided for DINOtxt segmentation")

        padded_image, padding_info = self.pad_image_to_square(image)

        if output_size is None:
            output_size = (padding_info.padded_height, padding_info.padded_width)

        image_tensor = self._preprocess_image(padded_image)

        text_features = self._encode_texts(text_prompts)

        with torch.autocast(self.device, dtype=torch.float16 if self.device == 'cuda' else torch.float32):
            with torch.no_grad():
                cls_tokens, _, patch_tokens = self.model.encode_image_with_patch_tokens(image_tensor)

        text_features_patch_aligned = text_features[:, 1024:].float()

        B, P, D = patch_tokens.shape
        H = W = int(P ** 0.5)  # Assuming square patch grid
        patch_features = patch_tokens.movedim(2, 1).unflatten(2, (H, W)).float()

        patch_features = F.interpolate(
            patch_features,
            size=output_size,
            mode="bicubic",
            align_corners=False
        )

        patch_features = F.normalize(patch_features, p=2, dim=1)  # [B, D, H, W]
        text_features_patch_aligned = F.normalize(text_features_patch_aligned, p=2, dim=1)  # [num_classes, D]
        per_patch_similarity = torch.einsum(
            "bdhw,cd->bchw",
            patch_features,
            text_features_patch_aligned
        )

        per_patch_similarity = per_patch_similarity.squeeze(0).cpu().numpy()  # [num_classes, H, W]
        segmentation_masks = []
        for idx, class_name in enumerate(text_prompts):
            similarity_map = per_patch_similarity[idx]  # [H, W]
            similarity_min = similarity_map.min()
            similarity_max = similarity_map.max()
            if similarity_max > similarity_min:
                confidence_map = (similarity_map - similarity_min) / (similarity_max - similarity_min)
            else:
                confidence_map = np.zeros_like(similarity_map)

            confidence = float(np.mean(np.sort(similarity_map.flatten())[-100:]))  # Top 100 patches
            seg_mask = SegmentationMask(
                mask=confidence_map,  # Continuous [0, 1] mask
                class_name=class_name,
                confidence=confidence,
                class_id=idx
            )
            segmentation_masks.append(seg_mask)

        if return_padding_info:
            return segmentation_masks, padding_info
        return segmentation_masks

    def predict_batch(
        self,
        images: List[np.ndarray],
        text_prompts: Optional[List[str]] = None,
        batch_size: int = 8,
        top_k: int = 5
    ) -> List[List[Classification]]:
        """
        Perform batch classification on multiple images.
        Processes images sequentially using predict() to avoid code duplication.

        Args:
            images: List of RGB images as numpy arrays
            text_prompts: Text descriptions of classes
            batch_size: Batch size for inference (unused, kept for interface compatibility)
            top_k: Return top-k predictions per image

        Returns:
            List of classification results, one per image
        """
        all_results = []

        for img in images:
            classifications = self.predict(img, text_prompts=text_prompts, top_k=top_k)
            all_results.append(classifications)

        return all_results

    def get_class_names(self) -> List[str]:
        """Get current text prompts (class names)."""
        return self.text_prompts

    def get_num_classes(self) -> int:
        """Get number of classes (text prompts)."""
        return len(self.text_prompts)

    def get_model_size_mb(self) -> float:
        """
        Get model size in MB.

        Returns:
            Model size in megabytes
        """
        try:
            num_params = sum(p.numel() for p in self.model.parameters())
            size_mb = (num_params * 4) / (1024 ** 2)
            return float(size_mb)
        except Exception as e:
            print(f"Warning: Could not calculate model size: {e}")
            return 0.0

    def __repr__(self) -> str:
        prompts_str = str(self.text_prompts[:3]) + "..." if len(self.text_prompts) > 3 else str(self.text_prompts)
        return (f"DINOtxtModel(model={self.model_name}, "
                f"device={self.device}, "
                f"threshold={self.confidence_threshold}, "
                f"prompts={prompts_str})")
