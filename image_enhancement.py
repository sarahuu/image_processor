import os
import subprocess
from pathlib import Path

class RealESRGANRunner:
    """
    A class to run Real-ESRGAN inference on a folder of images.
    """

    def __init__(self, model_name="RealESRGAN_x4plus", input_folder="images",
                 output_folder="results", outscale=3.5, face_enhance=True, device=None):
        self.model_name = model_name
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.outscale = outscale
        self.face_enhance = face_enhance
        self.device = device  # e.g., "cpu" or "cuda"

        # Create output folder if it doesn't exist
        self.output_folder.mkdir(exist_ok=True)

    def build_command(self):
        """
        Build the command line to run Real-ESRGAN inference.
        """
        cmd = [
            "python",
            "Real-ESRGAN/inference_realesrgan.py",
            "-n", self.model_name,
            "-i", str(self.input_folder),
            "-o", str(self.output_folder),
            "--outscale", str(self.outscale)
        ]
        if self.face_enhance:
            cmd.append("--face_enhance")
        if self.device:
            cmd.extend(["--device", self.device])
        return cmd

    def run(self):
        """
        Run the Real-ESRGAN inference.
        """
        cmd = self.build_command()
        print("Running command:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"Processing complete! Upscaled images are in '{self.output_folder}'.")


