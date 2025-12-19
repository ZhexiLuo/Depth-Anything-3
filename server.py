"""
Depth Anything 3 Server: HTTP API for monocular depth estimation.

Usage:
    python server.py --port 5003

API:
    POST /estimate
    Request:
        {
            "image_path": str,
            "output_dir": str,
            "output_name": str,
            "intrinsics": [[3x3 matrix]],
            "process_res": int (default: 504),
            "resize_to_original": bool (default: true),
            "save_depth": bool (default: true),
            "save_conf": bool (default: false),
            "save_depth_vis": bool (default: false),
            "save_glb": bool (default: false),
            "glb_conf_thresh_percentile": float (default: 40.0),
            "glb_num_max_points": int (default: 1000000),
            "glb_show_cameras": bool (default: true)
        }
    Response:
        {
            "status": "success" | "error",
            "message": str,
            "outputs": {"depth": str|null, "conf": str|null, "depth_vis": str|null, "glb": str|null}
        }
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image

from depth_anything_3.api import DepthAnything3


app = Flask(__name__)
model = None


def load_model():
    """Load DA3 model globally."""
    global model
    print("Loading DA3-Large model...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-Large").to("cuda")
    print("Model loaded successfully.")


@app.route('/estimate', methods=['POST'])
def estimate():
    """Estimate depth from single image."""
    try:
        data = request.json
        image_path = data["image_path"]
        output_dir = data["output_dir"]
        output_name = data["output_name"]
        intrinsics = np.array(data["intrinsics"], dtype=np.float32)

        # Processing parameters
        process_res = data.get("process_res", 504)
        resize_to_original = data.get("resize_to_original", True)

        # Save options
        save_depth = data.get("save_depth", True)
        save_conf = data.get("save_conf", False)
        save_depth_vis = data.get("save_depth_vis", False)
        save_glb = data.get("save_glb", False)

        # GLB parameters
        glb_conf_thresh_percentile = data.get("glb_conf_thresh_percentile", 40.0)
        glb_num_max_points = data.get("glb_num_max_points", 1000000)
        glb_show_cameras = data.get("glb_show_cameras", True)

        if not Path(image_path).exists():
            return jsonify({
                "status": "error",
                "message": f"Image not found: {image_path}"
            }), 400

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Get original image size for resize
        original_size = None
        if resize_to_original:
            img = Image.open(image_path)
            original_size = (img.height, img.width)  # (H, W)

        # Build export format
        export_formats = []
        if save_glb:
            export_formats.append("glb")
        if save_depth_vis:
            export_formats.append("depth_vis")

        export_format = "-".join(export_formats) if export_formats else None
        export_dir = output_dir if export_format else None

        # Inference
        prediction = model.inference(
            image=[image_path],
            intrinsics=intrinsics[None, ...],
            process_res=process_res,
            export_dir=export_dir,
            export_format=export_format,
            conf_thresh_percentile=glb_conf_thresh_percentile,
            num_max_points=glb_num_max_points,
            show_cameras=glb_show_cameras
        )

        outputs = {"depth": None, "conf": None, "depth_vis": None, "glb": None}

        # Get depth and conf from prediction
        depth = prediction.depth[0]
        conf = prediction.conf[0] if hasattr(prediction, 'conf') and prediction.conf is not None else None

        # Resize to original resolution if needed
        if resize_to_original and original_size is not None:
            target_h, target_w = original_size
            depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if conf is not None:
                conf = cv2.resize(conf, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Save depth.npy
        if save_depth:
            depth_path = Path(output_dir) / f"{output_name}_depth.npy"
            np.save(depth_path, depth.astype(np.float32))
            outputs["depth"] = str(depth_path)

        # Save conf.npy
        if save_conf and conf is not None:
            conf_path = Path(output_dir) / f"{output_name}_conf.npy"
            np.save(conf_path, conf.astype(np.float32))
            outputs["conf"] = str(conf_path)

        # Rename depth_vis output
        if save_depth_vis:
            # DA3 saves as "output_depth_vis.png"
            depth_vis_path = Path(output_dir) / "output_depth_vis.png"
            if depth_vis_path.exists():
                renamed_path = Path(output_dir) / f"{output_name}_depth_vis.png"
                depth_vis_path.rename(renamed_path)
                outputs["depth_vis"] = str(renamed_path)

        # Rename GLB output
        if save_glb:
            glb_path = Path(output_dir) / "output.glb"
            if glb_path.exists():
                renamed_glb = Path(output_dir) / f"{output_name}.glb"
                glb_path.rename(renamed_glb)
                outputs["glb"] = str(renamed_glb)

        return jsonify({
            "status": "success",
            "message": "Depth estimation completed",
            "outputs": outputs
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": "DA3-Large"})


def main():
    parser = argparse.ArgumentParser(description="Depth Anything 3 Server")
    parser.add_argument("--port", type=int, default=5003, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    args = parser.parse_args()

    load_model()

    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
