import argparse
from pathlib import Path
import cv2
import pandas as pd
import mediapipe as mp


def clamp_bbox(x0, y0, x1, y1, w, h):
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(1, min(w, x1))
    y1 = max(1, min(h, y1))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return x0, y0, x1, y1


def crop_with_margin(frame, box, margin=0.30):
    x0, y0, x1, y1 = box
    h, w = frame.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    mx, my = int(bw * margin), int(bh * margin)
    xx0 = max(0, x0 - mx)
    yy0 = max(0, y0 - my)
    xx1 = min(w, x1 + mx)
    yy1 = min(h, y1 + my)
    return frame[yy0:yy1, xx0:xx1]


def expand_bbox_for_forehead(x0, y0, x1, y1, left_eye_xy, right_eye_xy,
                             target_eye_y=0.38, max_extra_top=0.90):
    bw = x1 - x0
    bh = y1 - y0
    if bw <= 0 or bh <= 0:
        return x0, y0, x1, y1

    eye_y = 0.5 * (left_eye_xy[1] + right_eye_xy[1])
    num = eye_y - y0
    denom = (1.0 - target_eye_y)
    if denom <= 1e-6:
        return x0, y0, x1, y1

    t = (target_eye_y * bh - num) / denom
    if t <= 0:
        return x0, y0, x1, y1

    t = min(t, max_extra_top * bh)
    return x0, int(y0 - t), x1, y1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--save_size", type=int, default=224)
    ap.add_argument("--face_margin", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="Skip if output file already exists")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # detector for RAF/ExpW (but NOT for FERPlus 48x48 crops)
    fd = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    kept = []
    dropped = 0
    n = len(df) if args.limit <= 0 else min(len(df), args.limit)

    for i in range(n):
        r = df.iloc[i].to_dict()
        img_path = str(r["image"])
        dataset = str(r.get("dataset", "unk")).lower()
        split = str(r.get("split", "train")).lower()

        bgr = cv2.imread(img_path)
        if bgr is None:
            dropped += 1
            continue

        # output filename
        uid = str(r.get("uid", "")).strip()
        base = uid if uid else (Path(img_path).stem + f"_{i:07d}")
        out_path = out_dir / dataset / split / f"{base}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if args.resume and out_path.exists():
            r["orig_image"] = img_path
            r["image"] = str(out_path.resolve())
            r["proc_ok"] = 1
            r["proc_method"] = "resume"
            kept.append(r)
            continue

        # FERPlus: skip face detection (already aligned 48x48 crops)
        if dataset == "ferplus":
            face_img = bgr
            method = "ferplus_resize_only"
        else:
            h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = fd.process(rgb)

            face_img = None
            method = "detector"

            if res.detections:
                det = res.detections[0]
                bb = det.location_data.relative_bounding_box
                x0 = int(bb.xmin * w)
                y0 = int(bb.ymin * h)
                x1 = int((bb.xmin + bb.width) * w)
                y1 = int((bb.ymin + bb.height) * h)
                x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1, w, h)

                # forehead expansion using face_detection eye keypoints
                kps = det.location_data.relative_keypoints
                re = (kps[0].x * w, kps[0].y * h)
                le = (kps[1].x * w, kps[1].y * h)

                x0, y0, x1, y1 = expand_bbox_for_forehead(
                    x0, y0, x1, y1, left_eye_xy=le, right_eye_xy=re,
                    target_eye_y=0.38, max_extra_top=0.90
                )
                x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1, w, h)

                face_img = crop_with_margin(bgr, (x0, y0, x1, y1), margin=args.face_margin)

            # fallback: if detector fails, keep original
            if face_img is None or face_img.size == 0:
                face_img = bgr
                method = "fallback_full_image"

        face_img = cv2.resize(face_img, (args.save_size, args.save_size), interpolation=cv2.INTER_AREA)
        ok = cv2.imwrite(str(out_path), face_img)
        if not ok:
            dropped += 1
            continue

        r["orig_image"] = img_path
        r["image"] = str(out_path.resolve())
        r["proc_ok"] = 1
        r["proc_method"] = method
        kept.append(r)

        if (i + 1) % 1000 == 0:
            print(f"[{i+1}/{n}] kept={len(kept)} dropped={dropped}")

    fd.close()

    out = pd.DataFrame(kept)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print("[OK] wrote:", args.out_csv)
    print("[STATS] kept=", len(out), "dropped=", dropped)


if __name__ == "__main__":
    main()