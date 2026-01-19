"""
LTX-2 Video Generation Model - Replicate Predictor
Generates video from text prompts using Lightricks/LTX-2
"""

import os
import tempfile
from cog import BasePredictor, Input, Path

import torch

# Model paths (downloaded during setup)
MODEL_DIR = "/src/models/ltx-2"
# Using FP8 distilled for optimal speed on H100
CHECKPOINT_PATH = f"{MODEL_DIR}/ltx-2-19b-distilled-fp8.safetensors"
DISTILLED_LORA_PATH = f"{MODEL_DIR}/ltx-2-19b-distilled-lora-384.safetensors"
SPATIAL_UPSAMPLER_PATH = f"{MODEL_DIR}/ltx-2-spatial-upscaler-x2-1.0.safetensors"
# gemma_root should point to parent dir containing both text_encoder/ and tokenizer/
GEMMA_ROOT_PATH = MODEL_DIR


class Predictor(BasePredictor):
    def setup(self):
        """Download and load LTX-2 model into memory"""
        from huggingface_hub import snapshot_download

        # Download only the files we need (not the entire 314GB repo)
        print("Downloading LTX-2 model from HuggingFace...")
        snapshot_download(
            repo_id="Lightricks/LTX-2",
            local_dir=MODEL_DIR,
            allow_patterns=[
                "ltx-2-19b-distilled-fp8.safetensors",      # Main checkpoint (~27GB)
                "ltx-2-19b-distilled-lora-384.safetensors", # Distilled LoRA (~7.7GB)
                "ltx-2-spatial-upscaler-x2-1.0.safetensors", # Spatial upsampler (~1GB)
                "text_encoder/**",                          # Text encoder
                "tokenizer/**",                             # Tokenizer
            ],
        )
        print("Model downloaded successfully")

        # Initialize the pipeline using ltx_pipelines
        print("Loading LTX-2 pipeline...")
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps

        # Configure distilled LoRA for faster inference
        distilled_lora = [
            LoraPathStrengthAndSDOps(
                DISTILLED_LORA_PATH,
                0.6,
                LTXV_LORA_COMFY_RENAMING_MAP
            ),
        ]

        self.pipe = TI2VidTwoStagesPipeline(
            checkpoint_path=CHECKPOINT_PATH,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
            gemma_root=GEMMA_ROOT_PATH,
            loras=[],
            fp8transformer=True,  # Memory optimization for H100
        )
        print("Pipeline loaded successfully")

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt describing the video to generate",
            default="A woman with long brown hair looks around, her eyes sparkling with curiosity"
        ),
        width: int = Input(
            description="Video width in pixels",
            default=768,
            ge=256,
            le=1280
        ),
        height: int = Input(
            description="Video height in pixels",
            default=512,
            ge=256,
            le=720
        ),
        num_frames: int = Input(
            description="Number of frames to generate (more frames = longer video). Use 121 for ~5sec, 241 for ~10sec",
            default=241,
            ge=9,
            le=257
        ),
        frame_rate: float = Input(
            description="Frames per second for output video",
            default=25.0,
            ge=8.0,
            le=60.0
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps (more = better quality, slower)",
            default=40,
            ge=1,
            le=100
        ),
        guidance_scale: float = Input(
            description="CFG guidance scale - how closely to follow the prompt",
            default=3.0,
            ge=1.0,
            le=20.0
        ),
        seed: int = Input(
            description="Random seed for reproducibility (-1 for random)",
            default=-1
        ),
    ) -> Path:
        """Generate a video from text prompt"""
        import imageio
        import numpy as np

        # Handle seed
        if seed == -1:
            seed = int.from_bytes(os.urandom(4), "big")

        print(f"Generating video with seed: {seed}")
        print(f"Prompt: {prompt}")
        print(f"Resolution: {width}x{height}, Frames: {num_frames}, FPS: {frame_rate}")

        # Generate video frames
        result = self.pipe(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            cfg_guidance_scale=guidance_scale,
        )

        # Handle different return types from pipeline
        output_path = Path(tempfile.mktemp(suffix=".mp4"))

        if isinstance(result, str) and os.path.exists(result):
            # Pipeline returned a file path
            import shutil
            shutil.move(result, str(output_path))
        elif hasattr(result, 'frames'):
            # Pipeline returned an object with frames attribute
            frames = result.frames
            if isinstance(frames, torch.Tensor):
                frames = frames.cpu().numpy()
            # Normalize to uint8 if needed
            if frames.max() <= 1.0:
                frames = (frames * 255).astype(np.uint8)
            imageio.mimwrite(str(output_path), frames, fps=frame_rate)
        elif isinstance(result, (list, np.ndarray, torch.Tensor)):
            # Pipeline returned frames directly
            frames = result
            if isinstance(frames, torch.Tensor):
                frames = frames.cpu().numpy()
            if isinstance(frames, np.ndarray) and frames.max() <= 1.0:
                frames = (frames * 255).astype(np.uint8)
            imageio.mimwrite(str(output_path), frames, fps=frame_rate)
        else:
            # Try to find video attribute or save method
            if hasattr(result, 'video'):
                frames = result.video
                if isinstance(frames, torch.Tensor):
                    frames = frames.cpu().numpy()
                if frames.max() <= 1.0:
                    frames = (frames * 255).astype(np.uint8)
                imageio.mimwrite(str(output_path), frames, fps=frame_rate)
            elif hasattr(result, 'save'):
                result.save(str(output_path))
            else:
                raise ValueError(f"Unknown pipeline output type: {type(result)}")

        print(f"Video saved to: {output_path}")
        return output_path
