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
# Using FP8 distilled checkpoint for optimal speed on H100
CHECKPOINT_PATH = f"{MODEL_DIR}/ltx-2-19b-distilled-fp8.safetensors"
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
                "ltx-2-19b-distilled-fp8.safetensors",  # Main checkpoint (~27GB)
                "text_encoder/**",                      # Text encoder
                "tokenizer/**",                         # Tokenizer
            ],
        )
        print("Model downloaded successfully")

        # Initialize the pipeline using ltx_pipelines
        # Using DistilledPipeline - simplest and fastest, uses predefined sigmas
        print("Loading LTX-2 DistilledPipeline...")
        from ltx_pipelines.distilled import DistilledPipeline

        self.pipe = DistilledPipeline(
            checkpoint_path=CHECKPOINT_PATH,
            gemma_root=GEMMA_ROOT_PATH,
            spatial_upsampler_path=None,  # No upscaling to save memory
            loras=[],  # No custom LoRAs
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
            default=121,  # ~5 seconds, safer for memory
            ge=9,
            le=257
        ),
        frame_rate: float = Input(
            description="Frames per second for output video",
            default=25.0,
            ge=8.0,
            le=60.0
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

        # Generate video frames with memory optimization
        # DistilledPipeline uses predefined sigmas, no CFG needed
        # NOTE: DistilledPipeline does NOT accept negative_prompt (only TI2VidOneStagePipeline does)
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                images=[],  # Empty for text-to-video
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
            )

        # Clear CUDA cache to free memory before processing result
        torch.cuda.empty_cache()

        # DistilledPipeline returns tuple: (video_frames_iterator, audio_tensor)
        print(f"Pipeline returned type: {type(result)}")

        output_path = Path(tempfile.mktemp(suffix=".mp4"))

        # Handle tuple return (iterator, audio)
        if isinstance(result, tuple) and len(result) == 2:
            frames_iter, audio = result
            # Collect frames from iterator
            frames_list = []
            for frame_batch in frames_iter:
                if isinstance(frame_batch, torch.Tensor):
                    frame_batch = frame_batch.cpu().numpy()
                frames_list.append(frame_batch)

            if frames_list:
                frames = np.concatenate(frames_list, axis=0) if len(frames_list) > 1 else frames_list[0]
                # Normalize to uint8 if needed
                if frames.max() <= 1.0:
                    frames = (frames * 255).astype(np.uint8)
                # Ensure correct shape: (num_frames, height, width, channels)
                if frames.ndim == 4 and frames.shape[-1] != 3:
                    frames = np.transpose(frames, (0, 2, 3, 1))
                imageio.mimwrite(str(output_path), frames, fps=frame_rate)
            else:
                raise ValueError("Pipeline returned empty frames iterator")
        elif isinstance(result, torch.Tensor):
            # Direct tensor output
            frames = result.cpu().numpy()
            if frames.max() <= 1.0:
                frames = (frames * 255).astype(np.uint8)
            if frames.ndim == 4 and frames.shape[-1] != 3:
                frames = np.transpose(frames, (0, 2, 3, 1))
            imageio.mimwrite(str(output_path), frames, fps=frame_rate)
        else:
            raise ValueError(f"Unknown pipeline output type: {type(result)}")

        print(f"Video saved to: {output_path}")
        return output_path
