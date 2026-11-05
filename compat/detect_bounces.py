"""Detect ball bounces from an existing tracking CSV without rerunning TrackNet."""
import argparse
from pathlib import Path
from .bounce_detection import DEFAULT_MODEL, process_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True, help='Original or tracked video, with the same frame order as CSV')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1]/'outputs')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--threshold', type=float, default=0.55)
    parser.add_argument('--max-gap', type=int, default=2, help='Maximum missing source frames to interpolate')
    parser.add_argument('--min-separation', type=float, default=0.12, help='Minimum candidate event spacing in seconds')
    parser.add_argument('--model-fps', type=float, default=30.0)
    parser.add_argument('--reference-size', type=int, nargs=2, default=[1280,720], metavar=('WIDTH','HEIGHT'))
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    process_csv(args.csv, args.video, args.output_dir, overwrite=args.overwrite,
                render_video=not args.no_video, model_path=args.model, threshold=args.threshold,
                max_gap=args.max_gap, min_separation_seconds=args.min_separation,
                model_fps=args.model_fps, reference_width=args.reference_size[0],
                reference_height=args.reference_size[1])


if __name__ == '__main__':
    main()
