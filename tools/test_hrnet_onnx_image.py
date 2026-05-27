import os
import cv2
import argparse
import numpy as np
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test HRNet facial landmark ONNX model on a single image"
    )

    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="path to ONNX model"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="path to input image"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="onnx_result.jpg",
        help="path to save visualization"
    )

    parser.add_argument(
        "--input_w",
        type=int,
        default=256,
        help="model input width"
    )

    parser.add_argument(
        "--input_h",
        type=int,
        default=256,
        help="model input height"
    )

    parser.add_argument(
        "--norm",
        type=str,
        default="imagenet",
        choices=["imagenet", "none"],
        help="input normalization type"
    )

    parser.add_argument(
        "--draw_index",
        action="store_true",
        help="draw landmark index"
    )

    parser.add_argument(
        "--score_thresh",
        type=float,
        default=None,
        help="optional score threshold for drawing landmarks"
    )

    return parser.parse_args()


def preprocess_image(img_bgr, input_w=256, input_h=256, norm="imagenet"):
    """
    Args:
        img_bgr: original BGR image
        input_w: model input width
        input_h: model input height
        norm: imagenet or none

    Returns:
        img_resized: resized BGR image for visualization
        input_tensor: [1, 3, H, W]
    """
    img_resized = cv2.resize(img_bgr, (input_w, input_h))

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0

    if norm == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_rgb = (img_rgb - mean) / std

    img_chw = np.transpose(img_rgb, (2, 0, 1))
    input_tensor = img_chw[None, ...].astype(np.float32)

    return img_resized, input_tensor


def decode_heatmap_to_coords_and_scores(heatmap, input_w, input_h):
    """
    Decode heatmap to coordinates and confidence scores.

    Args:
        heatmap: [1, K, Hm, Wm] or [K, Hm, Wm]
        input_w: model input width
        input_h: model input height

    Returns:
        coords: [K, 2], coordinates in resized-image pixel coordinate system
        scores: [K], max heatmap response
    """
    if heatmap.ndim == 4:
        heatmap = heatmap[0]

    if heatmap.ndim != 3:
        raise ValueError(f"Unexpected heatmap shape: {heatmap.shape}")

    num_joints, hm_h, hm_w = heatmap.shape

    coords = np.zeros((num_joints, 2), dtype=np.float32)
    scores = np.zeros((num_joints,), dtype=np.float32)

    for i in range(num_joints):
        hm = heatmap[i]

        idx = int(np.argmax(hm))
        y, x = np.unravel_index(idx, hm.shape)

        px = x * (input_w - 1) / max(hm_w - 1, 1)
        py = y * (input_h - 1) / max(hm_h - 1, 1)

        coords[i, 0] = px
        coords[i, 1] = py
        scores[i] = float(hm[y, x])

    return coords, scores


def parse_onnx_outputs(outputs, output_names, input_w, input_h):
    """
    Compatible with three ONNX export styles:

    1. coords only:
        [1, K, 2] or [K, 2]

    2. heatmap only:
        [1, K, Hm, Wm] or [K, Hm, Wm]

    3. heatmap + coords + scores:
        heatmap: [1, K, Hm, Wm]
        coords : [1, K, 2]
        scores : [1, K]
    """
    heatmap = None
    coords = None
    scores = None

    # Case 1: three-output ONNX: heatmap, coords, scores
    if len(outputs) >= 3:
        for name, out in zip(output_names, outputs):
            lname = name.lower()

            if "heatmap" in lname:
                heatmap = out
            elif "coord" in lname:
                coords = out
            elif "score" in lname or "conf" in lname:
                scores = out

        # 如果输出名字不是 heatmap/coords/scores，也按 shape 判断
        if heatmap is None or coords is None:
            for out in outputs:
                if out.ndim == 4:
                    heatmap = out
                elif out.ndim == 3 and out.shape[-1] == 2:
                    coords = out
                elif out.ndim == 2:
                    scores = out

        if coords is None and heatmap is not None:
            coords, decoded_scores = decode_heatmap_to_coords_and_scores(
                heatmap,
                input_w,
                input_h
            )
            if scores is None:
                scores = decoded_scores

        else:
            if coords.ndim == 3:
                coords = coords[0]
            elif coords.ndim != 2:
                raise ValueError(f"Unexpected coords shape: {coords.shape}")

            coords = coords.astype(np.float32)

            if scores is not None:
                if scores.ndim == 2:
                    scores = scores[0]
                scores = scores.astype(np.float32)

            elif heatmap is not None:
                _, scores = decode_heatmap_to_coords_and_scores(
                    heatmap,
                    input_w,
                    input_h
                )

        return heatmap, coords, scores

    # Case 2: one-output ONNX
    output = outputs[0]

    if output.ndim == 4:
        heatmap = output
        coords, scores = decode_heatmap_to_coords_and_scores(
            heatmap,
            input_w,
            input_h
        )

    elif output.ndim == 3 and output.shape[-1] == 2:
        coords = output[0].astype(np.float32)
        scores = None

    elif output.ndim == 2 and output.shape[-1] == 2:
        coords = output.astype(np.float32)
        scores = None

    else:
        raise ValueError(f"Unsupported ONNX output shape: {output.shape}")

    return heatmap, coords, scores


def draw_landmarks(
    img,
    coords,
    scores=None,
    score_thresh=None,
    radius=2,
    draw_index=False
):
    """
    Draw landmarks on resized image.

    Args:
        img: resized BGR image
        coords: [K, 2]
        scores: [K] or None
        score_thresh: optional threshold
        radius: point radius
        draw_index: whether to draw landmark index
    """
    vis = img.copy()

    for i, (x, y) in enumerate(coords):
        if scores is not None and score_thresh is not None:
            if scores[i] < score_thresh:
                continue

        x = int(round(float(x)))
        y = int(round(float(y)))

        cv2.circle(vis, (x, y), radius, (0, 255, 0), -1)

        if draw_index:
            cv2.putText(
                vis,
                str(i),
                (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

    return vis


def draw_eye_score_info(img, scores):
    """
    For WFLW 98 landmarks, commonly:
        left eye : 60-67
        right eye: 68-75

    注意：
    如果你的项目里左右眼索引定义不同，要和 eye_region_detector.py 保持一致。
    """
    vis = img.copy()

    if scores is None:
        cv2.putText(
            vis,
            "scores: None",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )
        return vis

    if len(scores) < 76:
        cv2.putText(
            vis,
            f"scores length: {len(scores)}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )
        return vis

    left_eye_indices = list(range(60, 68))
    right_eye_indices = list(range(68, 76))

    left_eye_score = float(np.mean(scores[left_eye_indices]))
    right_eye_score = float(np.mean(scores[right_eye_indices]))

    selected_eye = "left" if left_eye_score >= right_eye_score else "right"

    text1 = f"left_eye_score : {left_eye_score:.4f}"
    text2 = f"right_eye_score: {right_eye_score:.4f}"
    text3 = f"selected_eye   : {selected_eye}"

    cv2.putText(
        vis,
        text1,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        vis,
        text2,
        (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        vis,
        text3,
        (10, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )

    return vis


def print_model_info(session):
    print("=" * 60)
    print("ONNX input info")
    print("=" * 60)
    for inp in session.get_inputs():
        print(f"name: {inp.name}, shape: {inp.shape}, type: {inp.type}")

    print("=" * 60)
    print("ONNX output info")
    print("=" * 60)
    for out in session.get_outputs():
        print(f"name: {out.name}, shape: {out.shape}, type: {out.type}")


def main():
    args = parse_args()

    if not os.path.exists(args.onnx):
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")

    img = cv2.imread(args.image)

    if img is None:
        raise ValueError(f"Failed to read image: {args.image}")

    vis_img, input_tensor = preprocess_image(
        img,
        input_w=args.input_w,
        input_h=args.input_h,
        norm=args.norm
    )

    session = ort.InferenceSession(
        args.onnx,
        providers=["CPUExecutionProvider"]
    )

    print_model_info(session)

    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]

    outputs = session.run(
        output_names,
        {
            input_name: input_tensor
        }
    )

    print("=" * 60)
    print("Runtime output shapes")
    print("=" * 60)
    for name, out in zip(output_names, outputs):
        print(f"{name}: {out.shape}")

    heatmap, coords, scores = parse_onnx_outputs(
        outputs=outputs,
        output_names=output_names,
        input_w=args.input_w,
        input_h=args.input_h
    )

    print("=" * 60)
    print("Parsed result")
    print("=" * 60)
    print("coords shape:", coords.shape)

    if heatmap is not None:
        print("heatmap shape:", heatmap.shape)

    if scores is not None:
        print("scores shape:", scores.shape)
        print("scores min/max/mean:",
              float(np.min(scores)),
              float(np.max(scores)),
              float(np.mean(scores)))
    else:
        print("scores: None")

    print("First 5 landmarks:")
    print(coords[:5])

    if scores is not None and len(scores) >= 76:
        left_eye_indices = list(range(60, 68))
        right_eye_indices = list(range(68, 76))

        left_eye_score = float(np.mean(scores[left_eye_indices]))
        right_eye_score = float(np.mean(scores[right_eye_indices]))

        selected_eye = "left" if left_eye_score >= right_eye_score else "right"

        print("=" * 60)
        print("Eye confidence test")
        print("=" * 60)
        print("left eye indices :", left_eye_indices)
        print("right eye indices:", right_eye_indices)
        print("left_eye_score  :", left_eye_score)
        print("right_eye_score :", right_eye_score)
        print("selected_eye    :", selected_eye)

    result = draw_landmarks(
        img=vis_img,
        coords=coords,
        scores=scores,
        score_thresh=args.score_thresh,
        radius=2,
        draw_index=args.draw_index
    )

    result = draw_eye_score_info(result, scores)

    success = cv2.imwrite(args.output, result)

    if not success:
        raise RuntimeError(f"Failed to save visualization to: {args.output}")

    print("=" * 60)
    print(f"Saved visualization to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()