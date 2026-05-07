import os
import json
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import threading

"""
This file provides a function to batch process scenes in the DL3DV dataset into required format.
Optimized for fast I/O with multi-threading and efficient image reading.
"""

# Try to import faster image reading libraries
try:
    from turbojpeg import TurboJPEG
    jpeg = TurboJPEG()
    USE_TURBOJPEG = True
    print("Using TurboJPEG for faster image decoding")
except ImportError:
    USE_TURBOJPEG = False
    print("TurboJPEG not available, using cv2.imread (slower)")

# Thread-local storage for cv2 operations (cv2 is not thread-safe for some operations)
thread_local = threading.local()


def fast_imread(file_path):
    """
    Fast image reading using TurboJPEG if available, otherwise cv2

    Args:
        file_path: path to image file

    Returns:
        image: numpy array in BGR format (OpenCV compatible)
    """
    if USE_TURBOJPEG:
        try:
            # Read file as binary
            with open(file_path, 'rb') as f:
                image_data = f.read()
            # Decode using TurboJPEG (3-5x faster than cv2)
            image = jpeg.decode(image_data, pixel_format=cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            # Fallback to cv2 if TurboJPEG fails
            pass

    # Standard cv2 imread
    return cv2.imread(file_path, cv2.IMREAD_COLOR)


def fast_imwrite(file_path, image, quality=95):
    """
    Fast image writing with optimization

    Args:
        file_path: output path
        image: numpy array
        quality: JPEG quality (1-100)
    """
    if USE_TURBOJPEG:
        try:
            # Encode using TurboJPEG
            encoded = jpeg.encode(image, quality=quality)
            with open(file_path, 'wb') as f:
                f.write(encoded)
            return True
        except Exception as e:
            pass

    # Fallback to cv2 with optimized parameters
    cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return True


def process_single_frame(frame_info, scene_path, images_undistort_dir,
                        fx_, fy_, cx_, cy_, w_, h_, distort):
    """
    Process a single frame (designed to be called in parallel)

    Returns:
        tuple: (success, frame_data) or (False, None)
    """
    try:
        frame, idx = frame_info
        file_name = frame['file_path'].split('/')[-1]
        input_file_path = os.path.join(scene_path, 'images_4', file_name)

        # Check if input image exists
        if not os.path.exists(input_file_path):
            return (False, f"Image not found: {input_file_path}")

        # Read image using fast imread
        image = fast_imread(input_file_path)
        if image is None:
            return (False, f"Failed to read: {input_file_path}")

        # Undistort
        h, w, _ = image.shape
        fx, fy, cx, cy = fx_/w_*w, fy_/h_*h, cx_/w_*w, cy_/h_*h
        intr = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        new_intr, roi = cv2.getOptimalNewCameraMatrix(intr, distort, (w, h), 0, (w, h))
        dst = cv2.undistort(image, intr, distort, None, new_intr)
        h, w, _ = dst.shape
        fx, fy, cx, cy = new_intr[0, 0], new_intr[1, 1], new_intr[0, 2], new_intr[1, 2]

        # Save undistorted image
        output_file_path = os.path.join(images_undistort_dir, file_name)
        fast_imwrite(output_file_path, dst)

        # Convert camera matrix
        c2w = np.asarray(frame["transform_matrix"])
        c2w[0:3, 1:3] *= -1
        c2w = c2w[[1, 0, 2, 3], :]
        c2w[2, :] *= -1
        w2c = np.linalg.inv(c2w)

        frame_new = {
            "file_path": f"images_undistort/{file_name}",
            "w2c": w2c.tolist(),
            "h": h,
            "w": w,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy
        }

        return (True, frame_new)

    except Exception as e:
        return (False, f"Error processing frame: {str(e)}")


def process_one_scene(scene_path, output_base_dir, num_workers=8):
    """
    Process one scene and save to corresponding output directory (with parallel processing)

    Args:
        scene_path: path to one scene folder (no trailing /)
        output_base_dir: base directory for output
        num_workers: number of threads for parallel processing

    Returns:
        success: True if processed successfully, False otherwise
    """
    try:
        scene_name = os.path.basename(scene_path)
        print(f"Processing scene: {scene_name}")

        # Create output directory for this scene
        scene_output_dir = os.path.join(output_base_dir, scene_name)
        os.makedirs(scene_output_dir, exist_ok=True)

        # Read input json
        json_file = os.path.join(scene_path, 'transforms.json')
        if not os.path.exists(json_file):
            print(f"  [SKIP] transforms.json not found in {scene_path}")
            return False

        json_data = json.load(open(json_file, 'r'))

        # Prepare output json
        new_json_file = os.path.join(scene_output_dir, 'opencv_cameras.json')
        new_data = {"scene_name": scene_name, "frames": []}

        # Get camera parameters
        w_ = json_data['w']
        h_ = json_data['h']
        fx_ = json_data['fl_x']
        fy_ = json_data['fl_y']
        cx_ = json_data['cx']
        cy_ = json_data['cy']
        k1, k2, p1, p2 = json_data['k1'], json_data['k2'], json_data['p1'], json_data['p2']
        distort = np.asarray([k1, k2, p1, p2])

        # Skip vertical videos
        if h_ > w_:
            print(f"  [SKIP] Vertical video: {scene_name} (h={h_}, w={w_})")
            return False

        num_frames = len(json_data['frames'])
        print(f"  Number of frames: {num_frames}")

        # Create undistort image folder for this scene
        images_undistort_dir = os.path.join(scene_output_dir, 'images_undistort')
        os.makedirs(images_undistort_dir, exist_ok=True)

        # Prepare frame processing function with fixed parameters
        process_func = partial(
            process_single_frame,
            scene_path=scene_path,
            images_undistort_dir=images_undistort_dir,
            fx_=fx_, fy_=fy_, cx_=cx_, cy_=cy_,
            w_=w_, h_=h_, distort=distort
        )

        # Process frames in parallel
        frame_infos = [(frame, idx) for idx, frame in enumerate(json_data['frames'])]
        results = [None] * num_frames  # Preserve order

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(process_func, frame_info): idx
                for idx, frame_info in enumerate(frame_infos)
            }

            # Collect results with progress bar
            with tqdm(total=num_frames, desc=f"  Processing frames", leave=False) as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        success, result = future.result()
                        if success:
                            results[idx] = result
                        else:
                            print(f"    [WARNING] {result}")
                    except Exception as e:
                        print(f"    [ERROR] Frame {idx}: {str(e)}")
                    pbar.update(1)

        # Add successfully processed frames to output (in original order)
        for result in results:
            if result is not None:
                new_data["frames"].append(result)

        # Save json file
        json.dump(new_data, open(new_json_file, 'w'), indent=4)
        print(f"  [SUCCESS] Saved to {scene_output_dir}")
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to process {scene_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def batch_process_scenes(input_dir, output_dir, max_scenes=None, num_workers=8):
    """
    Batch process all scenes in the input directory

    Args:
        input_dir: path to directory containing scene folders
        output_dir: base directory for output
        max_scenes: maximum number of scenes to process (None for all)
        num_workers: number of threads for parallel frame processing
    """
    # Get all scene directories
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    scene_dirs = []
    for item in os.listdir(input_dir):
        # import pdb;pdb.set_trace()
        item_path = os.path.join(input_dir, item) 
        if os.path.isdir(item_path):
            # Check if it has transforms.json
            if os.path.exists(os.path.join(item_path, 'transforms.json')):
                scene_dirs.append(item_path)
    scene_dirs = sorted(scene_dirs)

    if max_scenes is not None:
        scene_dirs = scene_dirs[:max_scenes]

    print(f"Found {len(scene_dirs)} scenes to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    # Create output base directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each scene
    success_count = 0
    failed_count = 0
    skipped_count = 0

    for i, scene_path in tqdm(enumerate(scene_dirs)):
        print(f"\n[{i+1}/{len(scene_dirs)}] Processing: {os.path.basename(scene_path)}")
        result = process_one_scene(scene_path, output_dir, num_workers=num_workers)

        if result is True:
            success_count += 1
        elif result is False:
            skipped_count += 1
        else:
            failed_count += 1

    # Summary
    print("\n" + "="*80)
    print("Processing Summary:")
    print(f"  Total scenes: {len(scene_dirs)}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Failed: {failed_count}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch process DL3DV dataset (Optimized with multi-threading and fast I/O)'
    )
    parser.add_argument('--input_dir', type=str,
                        help='Input directory containing scene folders')
    parser.add_argument('--output_dir', type=str,
                        help='Output base directory')
    parser.add_argument('--max_scenes', type=int, default=None,
                        help='Maximum number of scenes to process (default: all)')
    parser.add_argument('--single_scene', type=str, default=None,
                        help='Process a single scene by providing its full path')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of worker threads for parallel frame processing (default: 8)')

    args = parser.parse_args()

    print("="*80)
    print("DL3DV Dataset Processor - Optimized Version")
    print("="*80)
    print(f"Worker threads: {args.num_workers}")
    print(f"TurboJPEG: {'Available' if USE_TURBOJPEG else 'Not available (using cv2)'}")
    print("="*80)

    if args.single_scene:
        # Process single scene
        print("\nProcessing single scene mode")
        success = process_one_scene(args.single_scene, args.output_dir, num_workers=args.num_workers)
        if success:
            print("\nSingle scene processed successfully")
        else:
            print("\nFailed to process single scene")
    else:
        # Batch process
        batch_process_scenes(args.input_dir, args.output_dir, args.max_scenes, num_workers=args.num_workers)
