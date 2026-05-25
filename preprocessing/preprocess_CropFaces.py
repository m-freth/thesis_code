#!/usr/bin/env python3

## script to take bbox input and crop images to the face region, with defined margins and such, see input parameters to be specified
## NB: mostly copilot


import argparse
import csv
from pathlib import Path

from PIL import Image, ImageOps


def clamp(val, low, high):
    return max(low, min(val, high))


def make_crop_box(img_w, img_h, x, y, w, h, margin=0.2, square=False, top_extra=0.0):
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)

    if w <= 0 or h <= 0:
        raise ValueError(f"Non-positive bbox size: {(x, y, w, h)}")

    mx = w * margin
    my = h * margin
    top_pad = h * top_extra

    left = x - mx
    top = y - my - top_pad
    right = x + w + mx
    bottom = y + h + my

    if square:
        crop_w = right - left
        crop_h = bottom - top
        side = max(crop_w, crop_h)
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        left = cx - side / 2.0
        right = cx + side / 2.0
        top = cy - side / 2.0
        bottom = cy + side / 2.0

    left = clamp(int(round(left)), 0, img_w)
    top = clamp(int(round(top)), 0, img_h)
    right = clamp(int(round(right)), 0, img_w)
    bottom = clamp(int(round(bottom)), 0, img_h)

    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop after clamping: {(left, top, right, bottom)}")

    return left, top, right, bottom


def candidate_images(pil_img):
    """Try EXIF-corrected image first, then raw image as fallback."""
    img_t = ImageOps.exif_transpose(pil_img)
    yield "exif_transposed", img_t

    # Fallback: in some pipelines coordinates may refer to the raw pixel array.
    if pil_img.size != img_t.size:
        yield "raw", pil_img.copy()


def output_path_for_image(image_path: Path, output_dir: Path, input_root: Path | None):
    if input_root is not None:
        try:
            rel = image_path.resolve().relative_to(input_root.resolve())
            return output_dir / rel.with_suffix(".png")
        except ValueError:
            pass
    return output_dir / image_path.with_suffix(".png").name

def process_row(row, output_dir, input_root, margin, square, top_extra, resize):
    image_path = Path(row["image_path"])
    bbox_x = row["bbox_x"]
    bbox_y = row["bbox_y"]
    bbox_w = row["bbox_w"]
    bbox_h = row["bbox_h"]

    if not image_path.exists():
        print(f"[WARN] Missing image, skipping: {image_path}")
        return None

    try:
        with Image.open(image_path) as pil_img:
            pil_img.load()

            last_error = None
            for orientation_mode, img in candidate_images(pil_img):
                img = img.convert("RGB")
                img_w, img_h = img.size
                try:
                    crop_box = make_crop_box(
                        img_w=img_w,
                        img_h=img_h,
                        x=float(bbox_x),
                        y=float(bbox_y),
                        w=float(bbox_w),
                        h=float(bbox_h),
                        margin=margin,
                        square=square,
                        top_extra=top_extra,
                    )
                except Exception as e:
                    last_error = e
                    continue

                cropped = img.crop(crop_box)

                if resize is not None:
                    cropped = cropped.resize((resize, resize), Image.Resampling.LANCZOS)

                out_path = output_path_for_image(image_path, Path(output_dir), input_root)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cropped.save(out_path)

                return {
		    "image_path": str(image_path),
                    "output_path": str(out_path),
                    "crop_left": crop_box[0],
                    "crop_top": crop_box[1],
                    "crop_right": crop_box[2],
                    "crop_bottom": crop_box[3],
                    "orientation_mode": orientation_mode,
                    "status": "ok",
                }

            raise last_error if last_error else RuntimeError("No valid image orientation/crop candidate")

    except Exception as e:
        print(f"[WARN] Failed for {image_path}: {e}")
        return {
            "image_path": str(image_path),
            "output_path": "",
            "crop_left": "",
            "crop_top": "",
            "crop_right": "",
            "crop_bottom": "",
            "orientation_mode": "",
            "status": f"error: {e}",
        }


def main():
    parser = argparse.ArgumentParser(description="Crop face images in batch using OFIQ bounding boxes from CSV.")
    parser.add_argument("--csv", required=True, help="CSV with columns: image_path,bbox_x,bbox_y,bbox_w,bbox_h")
    parser.add_argument("--output-dir", required=True, help="Directory to save cropped images")
    parser.add_argument("--input-root", default=None, help="Input dataset root; preserves relative paths under this root")
    parser.add_argument("--margin", type=float, default=0.20, help="Fractional margin around bbox (default: 0.20)")
    parser.add_argument("--top-extra", type=float, default=0.15, help="Extra fraction of face height above bbox (default: 0.15)")
    parser.add_argument("--square", action="store_true", help="Force square crop")
    parser.add_argument("--resize", type=int, default=None, help="Optional output size, e.g. 512 gives 512x512")
    parser.add_argument("--report-csv", default=None, help="Optional CSV report of crop results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_root = Path(args.input_root).resolve() if args.input_root else None

    results = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"image_path", "bbox_x", "bbox_y", "bbox_w", "bbox_h"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

        for row in reader:
            if not all(row.get(k, "") != "" for k in required):
                results.append({
                    "image_path": row.get("image_path", ""),
                    "output_path": "",
                    "crop_left": "",
                    "crop_top": "",
                    "crop_right": "",
                    "crop_bottom": "",
                    "orientation_mode": "",
                    "status": "error: missing bbox values",
                })
                continue

            result = process_row(
                row=row,
                output_dir=output_dir,
                input_root=input_root,
                margin=args.margin,
                square=args.square,
                top_extra=args.top_extra,
                resize=args.resize,
            )
            if result is not None:
                results.append(result)

    if args.report_csv:
        with open(args.report_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_path",
                    "output_path",
                    "crop_left",
                    "crop_top",
                    "crop_right",
                    "crop_bottom",
                    "orientation_mode",
                    "status",
                ],
            )
            writer.writeheader()
            writer.writerows(results)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"[INFO] Done. Successfully cropped {ok_count}/{len(results)} images.")


if __name__ == "__main__":
    main()