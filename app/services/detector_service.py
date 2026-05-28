from pathlib import Path
from typing import Dict, List, Optional

import torch
from ultralytics import YOLO


class DetectorService:
    def __init__(self) -> None:
        self.model: Optional[YOLO] = None
        self.task_type: str = "detect"
        self.model_path: Optional[str] = None
        self.model_format: str = ""
        self.device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def names(self):
        return getattr(self.model, "names", {}) if self.model is not None else {}

    def load_model(self, model_path: str) -> None:
        model = YOLO(model_path, task="detect")
        self.model = model
        self.task_type = getattr(model, "task", "detect") or "detect"
        self.model_path = getattr(model, "ckpt_path", model_path)
        self.model_format = Path(model_path).suffix.lower()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def predict(self, image, conf: float = 0.25, iou: float = 0.45, imgsz: int = 416):
        if self.model is None:
            raise RuntimeError("模型尚未加载。")

        kwargs = dict(source=image, verbose=False, conf=conf, iou=iou, imgsz=imgsz)
        if self.model_format != ".engine":
            kwargs["device"] = self.device
        if self.device.startswith("cuda") and self.model_format == ".pt":
            kwargs["half"] = True
        if self.task_type == "obb":
            kwargs["task"] = "obb"

        results = self.model.predict(**kwargs)
        if not results:
            raise RuntimeError("模型未返回有效结果。")
        return results[0]

    def runtime_label(self) -> str:
        fmt = self.model_format.lstrip(".").upper() or "UNKNOWN"
        if self.model_format == ".engine":
            backend = "TensorRT"
        elif self.model_format == ".onnx":
            backend = f"ONNX/{self.device}"
        else:
            backend = self.device
        return f"{fmt} | {backend}"

    def render(self, result, show_labels: bool = True, show_conf: bool = True):
        return result.plot(labels=show_labels, conf=show_conf)

    def get_box_count(self, result) -> int:
        if self.task_type == "obb":
            return len(result.obb.xyxyxyxy) if result.obb is not None else 0
        return len(result.boxes.xyxy) if result.boxes is not None else 0

    def parse_rows(self, result) -> List[Dict]:
        rows: List[Dict] = []
        if self.task_type == "obb" and result.obb is not None:
            for idx, box in enumerate(result.obb, start=1):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                coords = box.xyxyxyxy[0].detach().cpu().tolist() if hasattr(box.xyxyxyxy, "detach") else box.xyxyxyxy[0].tolist()
                xs = [float(point[0]) for point in coords]
                ys = [float(point[1]) for point in coords]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                rows.append(
                    {
                        "index": idx,
                        "class_id": class_id,
                        "class_name": self.names.get(class_id, str(class_id)),
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                    }
                )
        elif result.boxes is not None:
            for idx, box in enumerate(result.boxes, start=1):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].detach().cpu().tolist() if hasattr(box.xyxy, "detach") else box.xyxy[0].tolist()
                x1, y1, x2, y2 = [float(value) for value in xyxy]
                rows.append(
                    {
                        "index": idx,
                        "class_id": class_id,
                        "class_name": self.names.get(class_id, str(class_id)),
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                    }
                )
        return rows

    def build_info_text(self, rows: List[Dict]) -> str:
        if not rows:
            return "未检测到目标"
        preview = rows[:12]
        lines = [f"类别: {row['class_name']}, 置信度: {row['confidence']:.2f}" for row in preview]
        if len(rows) > len(preview):
            lines.append(f"……其余 {len(rows) - len(preview)} 个目标已省略")
        return "\n".join(lines)
