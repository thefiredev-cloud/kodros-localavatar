"""SadTalker driver, run by the engine venv (numpy 1.23, torch 2.1) as a plain script.

Same steps as SadTalker's inference.py, but the three model groups load one at a time
and are freed before the next one, so peak VRAM stays near the largest single stage
(the GPU is shared with other services). Prints one ``ENGINE_STATS {json}`` line.

Imports nothing from the localavatar package (that venv has numpy 2).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vendor", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--driven_audio", required=True)
    p.add_argument("--source_image", required=True)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--expression_scale", type=float, default=1.0)
    p.add_argument("--pose_style", type=int, default=0)
    p.add_argument("--preprocess", default="full")
    p.add_argument("--still", action="store_true")
    p.add_argument("--enhancer", default=None)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()

    sys.path.insert(0, a.vendor)
    import torch
    from src.facerender.animate import AnimateFromCoeff
    from src.generate_batch import get_data
    from src.generate_facerender_batch import get_facerender_data
    from src.test_audio2coeff import Audio2Coeff
    from src.utils.init_path import init_path
    from src.utils.preprocess import CropAndExtract

    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    paths = init_path(a.checkpoint_dir, os.path.join(a.vendor, "src/config"), a.size, False, a.preprocess)
    work = os.path.join(a.result_dir, "work")
    os.makedirs(work, exist_ok=True)
    stats: dict = {"device": device, "stages": {}, "peak_alloc_mib": {}}

    def free(stage: str, t0: float) -> None:
        gc.collect()
        stats["stages"][stage] = round(time.monotonic() - t0, 2)
        if device == "cuda":
            stats["peak_alloc_mib"][stage] = round(torch.cuda.max_memory_allocated() / 2**20)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        t0 = time.monotonic()
        pre = CropAndExtract(paths, device)
        first_dir = os.path.join(work, "first_frame_dir")
        os.makedirs(first_dir, exist_ok=True)
        coeff, crop_pic, crop_info = pre.generate(a.source_image, first_dir, a.preprocess,
                                                  source_image_flag=True, pic_size=a.size)
        del pre
        free("face_3dmm", t0)
        if coeff is None:
            print("ENGINE_ERROR no face found in the photo", flush=True)
            return 3

        t0 = time.monotonic()
        a2c = Audio2Coeff(paths, device)
        batch = get_data(coeff, a.driven_audio, device, None, still=a.still)
        coeff_path = a2c.generate(batch, work, a.pose_style, None)
        del a2c, batch
        free("audio_to_motion", t0)

        t0 = time.monotonic()
        anim = AnimateFromCoeff(paths, device)
        data = get_facerender_data(coeff_path, crop_pic, coeff, a.driven_audio, a.batch_size, None, None, None,
                                   expression_scale=a.expression_scale, still_mode=a.still,
                                   preprocess=a.preprocess, size=a.size)
        result = anim.generate(data, work, a.source_image, crop_info, enhancer=a.enhancer,
                               background_enhancer=None, preprocess=a.preprocess, img_size=a.size)
        del anim, data
        free("render_paste_enhance", t0)

    final = os.path.join(a.result_dir, "talking.mp4")
    shutil.move(result, final)
    stats["video"] = final
    print("ENGINE_STATS " + json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
