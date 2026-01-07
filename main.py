from image_enhancement import RealESRGANRunner


if __name__ == "__main__":
    # Example usage
    runner = RealESRGANRunner(
        model_name="RealESRGAN_x4plus",
        input_folder="images",
        output_folder="results",
        outscale=3.5,
        face_enhance=True,
        device="cuda"  # or "cpu" if no GPU
    )
    runner.run()
