import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tests.helpers import generate_reference_figure


def refresh_reference_image():
    print("Refreshing reference image for visual tests...")

    # 1. Generate Plot
    fig = generate_reference_figure()

    # 2. Save to tests/reference_visual_output.png
    output_path = os.path.join(
        os.path.dirname(__file__), "../tests/reference_visual_output.png"
    )
    output_path = os.path.abspath(output_path)

    fig.savefig(output_path)

    print(f"Reference image saved to: {output_path}")


if __name__ == "__main__":
    refresh_reference_image()
