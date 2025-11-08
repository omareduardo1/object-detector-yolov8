import argparse, time
import cv2
from ultralytics import YOLO

def put_fps(frame, fps: float):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def parse_args():
    p = argparse.ArgumentParser(description="Real-Time Object Detector (YOLOv8)")
    p.add_argument("--model", type=str, default="yolov8n.pt",
                   help="model path or name (e.g., yolov8n.pt)")
    p.add_argument("--cam", type=str, default="0",
                   help="webcam index or video path (0,1... or file.mp4)")
    p.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    # performance / device
    p.add_argument("--device", type=str, default="mps", choices=("cpu","mps","cuda"),
                   help="inference device")
    p.add_argument("--imgsz", type=int, default=416, help="inference size (e.g., 320/416/512)")
    p.add_argument("--half", action="store_true", help="use FP16 (not for mps)")
    p.add_argument("--max_det", type=int, default=50, help="max detections per image")
    p.add_argument("--vid_stride", type=int, default=1, help="process every Nth frame")
    # webcam resolution (reduce for more FPS)
    p.add_argument("--width", type=int, default=640, help="camera width")
    p.add_argument("--height", type=int, default=480, help="camera height")
    return p.parse_args()

def open_capture(cam_arg: str):
    # Preferisci AVFoundation su macOS; fallback al default
    def _open_with_avfoundation(idx: int):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx)
        return cap

    if cam_arg.isdigit():
        idx = int(cam_arg)
        cap = _open_with_avfoundation(idx)
        if not cap.isOpened() and idx == 0:
            for alt in (1, 2):
                cap = _open_with_avfoundation(alt)
                if cap.isOpened():
                    return cap
        return cap
    else:
        return cv2.VideoCapture(cam_arg)

def main():
    args = parse_args()

    # YOLO scarica i pesi se mancanti
    model = YOLO(args.model)

    cap = open_capture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.cam}")

    # forza la risoluzione della webcam (aiuta molto gli FPS)
    if args.cam.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    t0 = time.time()
    frames = 0
    win_name = "Real-Time Object Detector (ESC for exit, +/- conf)"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # salta frame se richiesto
        if args.vid_stride > 1 and frames % args.vid_stride != 0:
            frames += 1
            cv2.imshow(win_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # mps non supporta half in modo stabile -> disabilita half su mps
        use_half = args.half and args.device != "mps"

        results = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            half=use_half,
            max_det=args.max_det,
            verbose=False
        )[0]

        annotated = results.plot()
        frames += 1
        fps = frames / max(time.time() - t0, 1e-6)
        annotated = put_fps(annotated, fps)

        cv2.imshow(win_name, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('+'), ord('=')):
            args.conf = min(args.conf + 0.05, 0.95)
        elif key == ord('-'):
            args.conf = max(args.conf - 0.05, 0.05)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()