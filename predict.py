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
TEXT_ENCODER_PATH = f"{MODEL_DIR}/text_encoder"


class Predictor(BasePredictor):
    def setup(self):
        """Download and load LTX-2 model into memory"""
        from huggingface_hub import snapshot_download

        # Download model weights from HuggingFace
        print("Downloading LTX-2 model from HuggingFace...")
        snapshot_download(
            repo_id="Lightricks/LTX-2",
            local_dir=MODEL_DIR,
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
            gemma_root=TEXT_ENCODER_PATH,
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

        # Handle seed
        if seed == -1:
            seed = int.from_bytes(os.urandom(4), "big")

        print(f"Generating video with seed: {seed}")
        print(f"Prompt: {prompt}")
        print(f"Resolution: {width}x{height}, Frames: {num_frames}, FPS: {frame_rate}")

        # Create output path
        output_path = Path(tempfile.mktemp(suffix=".mp4"))

        # Generate video
        self.pipe(
            prompt=prompt,
            output_path=str(output_path),
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            cfg_guidance_scale=guidance_scale,
        )

        print(f"Video saved to: {output_path}")
        return output_path
