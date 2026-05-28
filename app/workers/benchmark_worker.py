import statistics
import time
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from app.services.detector_service import DetectorService
from app.workers.detection_worker import DetectionMode

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for non-PT runtimes
    torch = None


class BenchmarkWorker(QThread):
    status_changed = pyqtSignal(str)
    benchmark_finished = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        detector: DetectorService,
        mode: DetectionMode,
        source: Union[str, Sequence[str]],
        sizes: Sequence[int],
        sample_limit: int,
        conf: float,
        iou: float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.detector = detector
        self.mode = mode
        self.source = source
        self.sizes = list(sizes)
        self.sample_limit = max(1, int(sample_limit))
        self.conf = conf
        self.iou = iou
        self._running = True

    def stop(self) -> None:
        self._running = False

    def _sync_cuda(self) -> None:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _load_samples(self) -> List[Tuple[str, object]]:
        samples: List[Tuple[str, object]] = []

        if self.mode == DetectionMode.IMAGE:
            image_path = str(self.source)
            frame = cv2.imread(image_path)
            if frame is not None:
                samples.append((Path(image_path).name, frame))
            return samples

        if self.mode == DetectionMode.FOLDER:
            for image_path in list(self.source)[: self.sample_limit]:
                frame = cv2.imread(str(image_path))
                if frame is not None:
                    samples.append((Path(str(image_path)).name, frame))
            return samples

        if self.mode == DetectionMode.VIDEO:
            capture = cv2.VideoCapture(str(self.source))
            try:
                index = 0
                while self._running and index < self.sample_limit:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    index += 1
                    samples.append((f"frame_{index}", frame))
            finally:
                capture.release()
            return samples

        return samples

    def run(self) -> None:
        try:
            samples = self._load_samples()
            if not samples:
                raise RuntimeError("没有可用于参数对比的样本。")

            results = []
            for size in self.sizes:
                if not self._running:
                    break

                self.status_changed.emit(f"正在测试输入尺寸 {size}，样本数 {len(samples)}")
                warmup_frame = samples[0][1]
                for _ in range(2):
                    self.detector.predict(warmup_frame, conf=self.conf, iou=self.iou, imgsz=size)

                confidences = []
                box_total = 0
                self._sync_cuda()
                start = time.perf_counter()
                for _, frame in samples:
                    if not self._running:
                        break
                    result = self.detector.predict(frame, conf=self.conf, iou=self.iou, imgsz=size)
                    rows = self.detector.parse_rows(result)
                    box_total += self.detector.get_box_count(result)
                    confidences.extend(float(row["confidence"]) for row in rows)
                self._sync_cuda()
                elapsed = max(time.perf_counter() - start, 1e-9)

                fps = len(samples) / elapsed
                avg_ms = elapsed * 1000.0 / len(samples)
                avg_conf = statistics.mean(confidences) if confidences else 0.0
                results.append(
                    {
                        "imgsz": size,
                        "samples": len(samples),
                        "elapsed_s": elapsed,
                        "fps": fps,
                        "avg_ms": avg_ms,
                        "boxes": box_total,
                        "avg_conf": avg_conf,
                        "runtime": self.detector.runtime_label(),
                    }
                )

            self.benchmark_finished.emit(results)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
