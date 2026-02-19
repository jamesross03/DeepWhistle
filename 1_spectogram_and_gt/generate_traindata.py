#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generates spectrogram images from WAV files and corresponding BIN
annotations. This is a refactored version of original script written by Pu Li
(2018).

@author: Pu Li
Refactoring: James Ross
"""

import argparse
from pathlib import Path
import os
import sys
import utils.wav2spec as wav2spec

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments."""

    parser = argparse.ArgumentParser(description="Generate spectrogram images")

    parser.add_argument(
        "--audio_dir", type=str, required=True, help="the path containing .wav files"
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        required=True,
        help="the path containing .bin files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="the path to output images"
    )

    parser.add_argument(
        "--frame_time_span",
        type=int,
        default=8,
        help="ms, length of time for one time window for dft",
    )
    parser.add_argument(
        "--step_time_span",
        type=int,
        default=2,
        help="ms, length of time step for spectrogram",
    )
    parser.add_argument(
        "--spec_clip_min",
        type=float,
        default=0,
        help="log magnitude spectrogram min-max normalization, minimum value",
    )
    parser.add_argument(
        "--spec_clip_max",
        type=float,
        default=6,
        help="log magnitude spectrogram min-max normalization, maximum value",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=5000,
        help="Hz, lower bound of frequency for spectrogram",
    )
    parser.add_argument(
        "--max_freq",
        type=int,
        default=50000,
        help="Hz, upper bound of frequency for spectrogram",
    )
    parser.add_argument(
        "--split_time",
        type=int,
        default=3000,
        help="ms, length of time for each output spectrogram image.",
    )

    return parser.parse_args()


def main() -> None:
    config = parse_args()

    audio_dir = Path(config.audio_dir)
    anno_dir = Path(config.annotation_dir)
    output_dir = Path(config.output_dir)

    wav_files = wav2spec.find_wav_files(str(audio_dir))
    wav_file_dict = {Path(f).name: f for f in wav_files}

    bin_files = wav2spec.findfiles(str(anno_dir), fnmatchex="*.bin")

    # find all .wav files that have corresponding .bin files.
    anno_wav_filenames = list(map(wav2spec.bin2wav_filename, bin_files))
    try:
        anno_wav_files = [wav_file_dict[filename] for filename in anno_wav_filenames]
    except KeyError as e:
        raise RuntimeError(f"Missing WAV file for annotation: {e}") from None

    # Checks all wav files are readable
    list(map(wav2spec.get_wav_samplewidth, anno_wav_files))

    # Check output dir exists
    wav2spec.check_dir(str(output_dir))

    total = len(anno_wav_files)
    # Process files
    for i, (wav_file, bin_file) in enumerate(zip(anno_wav_files, bin_files), start=1):
        print(f"Processing {i}/{total}): {wav_file}")

        wav_filename = Path(wav_file).stem
        output_path = output_dir / wav_filename
        wav2spec.check_dir(output_path)

        block_func = wav2spec.processBlock_lineGT

        output_func = wav2spec.log_magnitute_spectrum_GT_DCL_blockwise3(
            block_func,
            frame_time_span=config.frame_time_span,
            step_time_span=config.step_time_span,
            imsave_output_dir=str(output_path),
            min_freq=config.min_freq,
            max_freq=config.max_freq,
            split_time=config.split_time,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
        )

        count = output_func([wav_file, bin_file])
        print(f"Generated {count} images")


if __name__ == "__main__":
    main()
