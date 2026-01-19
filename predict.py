"""
LTX Video Generation Model - Replicate Predictor
Uses Diffusers for stable video generation
"""

import os
import tempfile
import gc
from cog import BasePredictor, Input, Path

import torch


class Predictor(BasePredictor):
    def setup(self):
        """Load LTX-Video model using Diffusers"""
        from diffusers import LTXPipeline

        print("Loading LTX-Video pipeline from Diffusers...")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Load pipeline - keep it simple, no offloading
        self.pipe = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
        )

        # Convert VAE to float32 to avoid numerical issues during decode
        print("Converting VAE to float32...")
        self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)

        self.pipe.to("cuda")

        # Enable VAE optimizations
        self.pipe.vae.enable_tiling()

        print("Pipeline loaded successfully")

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt describing the video to generate",
            default="A woman with long brown hair looks around, her eyes sparkling with curiosity"
        ),
        width: int = Input(
            description="Video width in pixels (must be divisible by 32)",
            default=512,
            ge=256,
            le=768
        ),
        height: int = Input(
            description="Video height in pixels (must be divisible by 32)",
            default=320,
            ge=256,
            le=512
        ),
        num_frames: int = Input(
            description="Number of frames to generate (must be 8n+1). Use 49 for ~2sec, 97 for ~4sec",
            default=49,
            ge=9,
            le=97
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

        # Print memory before generation
        print(f"GPU memory before: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Clear memory before generation
        gc.collect()
        torch.cuda.empty_cache()

        print("Starting generation...")
        negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

        try:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            print("Generation complete!")
            print(f"Result type: {type(result)}")
            print(f"Frames type: {type(result.frames)}")
            print(f"Frames[0] type: {type(result.frames[0])}")

            video = result.frames[0]
            print(f"Video has {len(video)} frames")

        except Exception as e:
            print(f"Error during generation: {type(e).__name__}: {e}")
            raise

        # Clear CUDA cache
        gc.collect()
        torch.cuda.empty_cache()

        # Export to video file
        output_path = Path(tempfile.mktemp(suffix=".mp4"))
        print(f"Exporting to {output_path}...")
        export_to_video(video, str(output_path), fps=24)

        print(f"Video saved to: {output_path}")
        return output_path
