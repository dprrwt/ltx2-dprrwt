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

        # Use sequential CPU offload - moves only active component to GPU
        # This is more aggressive memory management
        self.pipe.enable_sequential_cpu_offload()

        # Enable VAE optimizations
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()

        print("Pipeline loaded successfully")

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt describing the video to generate",
            default="A woman with long brown hair looks around, her eyes sparkling with curiosity"
        ),
        width: int = Input(
            description="Video width in pixels (must be divisible by 32)",
            default=512,  # Reduced for memory safety
            ge=256,
            le=768
        ),
        height: int = Input(
            description="Video height in pixels (must be divisible by 32)",
            default=320,  # Reduced for memory safety
            ge=256,
            le=512
        ),
        num_frames: int = Input(
            description="Number of frames to generate (must be 8n+1). Use 49 for ~2sec, 97 for ~4sec",
            default=49,  # Reduced for memory safety
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

        generator = torch.Generator().manual_seed(seed)  # CPU generator for offload compatibility

        # Generate video
        negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

        # Clear memory before generation
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        print("Starting generation (latent output)...")
        # First get latents to isolate the issue
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="latent",  # Skip VAE decode in pipeline
        )
        print("Diffusion complete, got latents")
        latents = result.frames
        print(f"Latent shape: {latents.shape}")

        # Clear memory before VAE decode
        gc.collect()
        torch.cuda.empty_cache()

        # Manually decode with VAE
        print("Decoding latents with VAE...")
        with torch.no_grad():
            # Move VAE to GPU explicitly
            self.pipe.vae.to("cuda")
            video_tensor = self.pipe.vae.decode(latents.to("cuda")).sample
            print(f"Decoded tensor shape: {video_tensor.shape}")

        # Convert to frames
        video_tensor = video_tensor.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # Process frames for export
        print("Processing frames...")
        video_tensor = (video_tensor / 2 + 0.5).clamp(0, 1)
        video_tensor = video_tensor.permute(0, 2, 3, 4, 1)  # B, F, H, W, C
        video_np = (video_tensor[0] * 255).numpy().astype("uint8")
        video = [frame for frame in video_np]
        print(f"Processed {len(video)} frames")

        # Export to video file
        output_path = Path(tempfile.mktemp(suffix=".mp4"))
        export_to_video(video, str(output_path), fps=24)

        print(f"Video saved to: {output_path}")
        return output_path
