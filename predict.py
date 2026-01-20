"""
LTX-Video Generation Model - Replicate Predictor
Weights baked into container for fast cold boot
"""

import os
import tempfile
from cog import BasePredictor, Input, Path

import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

# Path where weights are baked into container (downloaded during cog build)
MODEL_PATH = "/src/models/ltx-video"


class Predictor(BasePredictor):
    def setup(self):
        """Load LTX-Video model from local path (no download needed)"""
        print("=" * 50)
        print("LTX-Video Setup - Loading from baked weights")
        print("=" * 50)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        print(f"Loading model from: {MODEL_PATH}")

        # Load from local path (baked into container)
        self.pipe = LTXPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            local_files_only=True,  # Don't try to download, use local only
        )
        self.pipe.to("cuda")

        print("Pipeline loaded successfully!")
        print("=" * 50)

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt describing the video to generate",
            default="A woman with long brown hair looks around, her eyes sparkling with curiosity"
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="worst quality, inconsistent motion, blurry, jittery, distorted"
        ),
        width: int = Input(
            description="Video width (multiple of 32)",
            default=704,
            ge=256,
            le=1280
        ),
        height: int = Input(
            description="Video height (multiple of 32)",
            default=480,
            ge=256,
            le=720
        ),
        num_frames: int = Input(
            description="Number of frames (97, 129, 161, 193, 225, 257)",
            default=97,
            ge=33,
            le=257
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            default=50,
            ge=10,
            le=100
        ),
        guidance_scale: float = Input(
            description="Guidance scale",
            default=3.0,
            ge=1.0,
            le=10.0
        ),
        fps: int = Input(
            description="Frames per second for output video",
            default=24,
            ge=8,
            le=60
        ),
        seed: int = Input(
            description="Random seed (-1 for random)",
            default=-1
        ),
    ) -> Path:
        """Generate a video from text prompt"""
        if seed == -1:
            seed = int.from_bytes(os.urandom(4), "big")

        # Ensure dimensions are multiples of 32
        width = (width // 32) * 32
        height = (height // 32) * 32

        print(f"Generating video...")
        print(f"Seed: {seed}")
        print(f"Prompt: {prompt}")
        print(f"Resolution: {width}x{height}, Frames: {num_frames}")
        print(f"Steps: {num_inference_steps}, CFG: {guidance_scale}")

        generator = torch.Generator(device="cuda").manual_seed(seed)

        video = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]

        output_path = Path(tempfile.mktemp(suffix=".mp4"))
        export_to_video(video, str(output_path), fps=fps)

        print(f"Video saved: {output_path}")
        return output_path
