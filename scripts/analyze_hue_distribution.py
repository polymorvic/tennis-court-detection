import tyro
from tennis_court_detection.training.dataset_preparation import analyze_hue_distribution
from pathlib import Path


def run(
    images_path: Path,
    hist_output_dir: Path,
    fig_name: str = 'hue_distribution.png',
    hist_bins_num: int = 180
):
    '''
    uv run python scripts/analyze_hue_distribution.py --images-path data/pics --hist-output-dir results/court_color_distribution --fig-name hue_dist
    '''
    analyze_hue_distribution(images_path, hist_output_dir, fig_name, hist_bins_num)


if __name__ == "__main__":
    tyro.cli(run)
