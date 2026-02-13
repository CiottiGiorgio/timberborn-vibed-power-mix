import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tests.helpers import generate_reference_figure

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def refresh_reference_image():
    logger.info("Refreshing reference image for visual tests...")

    # 1. Generate Plot
    fig = generate_reference_figure()

    # 2. Save to tests/reference_visual_output.png
    output_path = os.path.join(
        os.path.dirname(__file__), "../tests/reference_visual_output.png"
    )
    output_path = os.path.abspath(output_path)

    fig.savefig(output_path)

    logger.info(f"Reference image saved to: {output_path}")


if __name__ == "__main__":
    refresh_reference_image()
