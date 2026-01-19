"""
LTX Video Generation Model - Replicate Predictor
Uses Diffusers for stable video generation
"""

import os
import tempfile
from cog import BasePredictor, Input, Path

import torch


class Predictor(BasePredictor):
    def setup(self):
        """Load LTX-Video model using Diffusers"""
        from diffusers import LTXPipeline

        print("Loading LTX-Video pipeline from Diffusers...")

        # Load with bfloat16 for H100 optimization
        self.pipe = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
        )
        self.pipe.to("cuda")

        # Enable memory optimizations
        self.pipe.enable_model_cpu_offload()

        print("Pipeline loaded successfully")

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt describing the video to generate",
            default="A woman with long brown hair looks around, her eyes sparkling with curiosity"
        ),
        width: int = Input(
            description="Video width in pixels (must be divisible by 32)",
            default=704,
            ge=256,
            le=1280
        ),
        height: int = Input(
            description="Video height in pixels (must be divisible by 32)",
            default=480,
            ge=256,
            le=720
        ),
        num_frames: int = Input(
            description="Number of frames to generate (must be 8n+1). Use 97 for ~4sec, 161 for ~6.5sec",
            default=97,
            ge=9,
            le=161
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps (more = better quality but slower)",
            default=30,
            ge=10,
            le=50
        ),
        guidance_scale: float = Input(
            description="Guidance scale for CFG (higher = more prompt adherence)",
            default=3.0,
            ge=1.0,
            le=10.0
        ),
        seed: int = Input(
            description="Random seed for reproducibility (-1 for random)",
            default=-1
        ),
    ) -> Path:
        """Generate a video from text prompt"""
        from diffusers.utils import export_to_video

        # Handle seed
        if seed == -1:
            seed = int.from_bytes(os.urandom(4), "big")

        print(f"Generating video with seed: {seed}")
        print(f"Prompt: {prompt}")
        print(f"Resolution: {width}x{height}, Frames: {num_frames}")
        print(f"Steps: {num_inference_steps}, CFG: {guidance_scale}")

        # Ensure dimensions are divisible by 32
        width = (width // 32) * 32
        height = (height // 32) * 32

        # Ensure frames follow 8n+1 pattern
        num_frames = ((num_frames - 1) // 8) * 8 + 1

        generator = torch.Generator("cuda").manual_seed(seed)

        # Generate video
        negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

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

        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Export to video file
        output_path = Path(tempfile.mktemp(suffix=".mp4"))
        export_to_video(video, str(output_path), fps=24)

        print(f"Video saved to: {output_path}")
        return output_path
