import cv2
import numpy as np


def flat_field_correction(image_path, output_path, blur_kernel_size=501):
    print(f"Processing {image_path}...")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Convert to floating point for division
    img_float = img.astype(np.float32)

    # Estimate illumination background using a large Gaussian blur
    # The kernel size must be odd.
    # Adjust blur_kernel_size significantly strictly larger than objects of interest but smaller than lighting variation
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    print("Estimating background illumination...")
    background = cv2.GaussianBlur(img_float, (blur_kernel_size, blur_kernel_size), 0)

    # Avoid division by zero
    background[background == 0] = 1

    # Calculate the mean of the background to maintain global brightness
    mean_background = np.mean(background)

    # Perform flat field correction: Image / Background * Mean
    print("Correcting image...")
    corrected_float = (img_float / background) * mean_background

    # Clip values to valid range [0, 255] and convert back to uint8
    corrected_float = np.clip(corrected_float, 0, 255)
    corrected_img = corrected_float.astype(np.uint8)

    # Save output
    cv2.imwrite(output_path, corrected_img)
    print(f"Saved corrected image to {output_path}")


if __name__ == "__main__":
    # Use raw string for paths or forward slashes to avoid escape character issues
    input_file = r"c:\work\cellpose-img\11b.JPG"
    output_file = r"c:\work\cellpose-img\11b_corrected.jpg"

    flat_field_correction(input_file, output_file)
