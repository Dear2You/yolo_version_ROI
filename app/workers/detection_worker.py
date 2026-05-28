import os
import time
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from app.services.detector_service import DetectorService


class DetectionMode(Enum):
    NONE = auto()
    IMAGE = auto()
    FOLDER = auto()
    VIDEO = auto()
    CAMERA = auto()


class DetectionWorker(QThread):
    frame_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    detection_finished = pyqtSignal(str)
    recording_ready = pyqtSignal(str)

    def __init__(
        self,
        detector: DetectorService,
        mode: DetectionMode,
        source: Union[int, str, Sequence[str]],
        conf: float,
        iou: float,
        imgsz: int,
        show_labels: bool,
        show_conf: bool,
        display_fps: int = 30,
        stream_emit_interval: int = 3,
        folder_emit_interval: int = 5,
        write_video_result: bool = False,
        save_folder_results: bool = False,
        temp_video_path: Optional[str] = None,
        temp_folder_output_dir: Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.detector = detector
        self.mode = mode
        self.source = source
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.show_labels = show_labels
        self.show_conf = show_conf
        self.display_fps = max(1, display_fps)
        self.stream_emit_interval = max(1, stream_emit_interval)
        self.folder_emit_interval = max(1, folder_emit_interval)
        self.write_video_result = write_video_result
        self.save_folder_results = save_folder_results
        self.temp_video_path = temp_video_path
        self.temp_folder_output_dir = temp_folder_output_dir

        self._running = True
        self._paused = False
        self._camera_recording = False
        self._camera_record_path: Optional[str] = None
        self._camera_writer = None
        self._fps = 0.0
        self._last_tick = None
        self._start_tick = None

    def stop(self) -> None:
        self._running = False
        self._paused = False

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def start_camera_recording(self, temp_path: str) -> None:
        self._camera_record_path = temp_path
        self._camera_recording = True
        self.status_changed.emit("开始录制摄像头检测结果")

    def stop_camera_recording(self) -> None:
        self._camera_recording = False
        if self._camera_writer is not None:
            self._camera_writer.release()
            self._camera_writer = None
        if self._camera_record_path and os.path.exists(self._camera_record_path):
            self.recording_ready.emit(self._camera_record_path)
        self.status_changed.emit("摄像头录制已停止")

    def _wait_if_paused(self) -> bool:
        while self._running and self._paused:
            self.msleep(80)
        return self._running

    def _update_fps(self) -> float:
        now = time.perf_counter()
        if self._start_tick is None:
            self._start_tick = now
        if self._last_tick is not None:
            dt = max(now - self._last_tick, 1e-6)
            instant = 1.0 / dt
            self._fps = instant if self._fps == 0 else (0.85 * self._fps + 0.15 * instant)
        self._last_tick = now
        return self._fps

    def _ensure_camera_writer(self, frame) -> None:
        if not self._camera_recording or self._camera_writer is not None or not self._camera_record_path:
            return
        h, w = frame.shape[:2]
        self._camera_writer = cv2.VideoWriter(
            self._camera_record_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.display_fps,
            (w, h),
        )

    def _process_frame(
        self,
        frame,
        source_name: str,
        frame_index: int,
        progress_current: int = 0,
        progress_total: int = 0,
        folder_output_path: Optional[str] = None,
        emit_payload: bool = True,
        need_annotated: bool = True,
    ):
        result = self.detector.predict(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz)
        fps = self._update_fps()

        if not emit_payload and not folder_output_path and not need_annotated:
            return None

        annotated = self.detector.render(result, show_labels=self.show_labels, show_conf=self.show_conf)

        if folder_output_path:
            cv2.imwrite(folder_output_path, annotated)

        if self.mode == DetectionMode.CAMERA and self._camera_recording:
            self._ensure_camera_writer(annotated)
            if self._camera_writer is not None:
                self._camera_writer.write(annotated)

        if not emit_payload:
            return annotated

        rows = self.detector.parse_rows(result)
        box_count = self.detector.get_box_count(result)
        info_text = self.detector.build_info_text(rows)

        payload = {
            "original": frame,
            "annotated": annotated,
            "rows": rows,
            "box_count": box_count,
            "info_text": info_text,
            "source_name": source_name,
            "frame_index": frame_index,
            "fps": fps,
            "progress_current": progress_current,
            "progress_total": progress_total,
        }
        self.frame_ready.emit(payload)
        return annotated

    def _run_folder(self) -> None:
        image_paths = list(self.source)
        if not image_paths:
            self.error_occurred.emit("所选文件夹中没有可检测的图片。")
            return

        if self.save_folder_results and self.temp_folder_output_dir:
            os.makedirs(self.temp_folder_output_dir, exist_ok=True)
        total = len(image_paths)
        for index, image_path in enumerate(image_paths, start=1):
            if not self._running:
                break
            if not self._wait_if_paused():
                break

            image = cv2.imread(image_path)
            if image is None:
                self.status_changed.emit(f"跳过无法读取的图片: {os.path.basename(image_path)}")
                continue

            output_path = None
            if self.save_folder_results and self.temp_folder_output_dir:
                output_path = os.path.join(self.temp_folder_output_dir, os.path.basename(image_path))
            emit_payload = index == 1 or index % self.folder_emit_interval == 0 or index == total
            if emit_payload:
                self.status_changed.emit(f"正在处理: {os.path.basename(image_path)} ({index}/{total})")
            self._process_frame(
                image,
                source_name=os.path.basename(image_path),
                frame_index=index,
                progress_current=index,
                progress_total=total,
                folder_output_path=output_path,
                emit_payload=emit_payload,
                need_annotated=emit_payload,
            )

        if self._running:
            self.detection_finished.emit("文件夹检测完成")

    def _is_remote_stream(self) -> bool:
        if not isinstance(self.source, str):
            return False
        return self.source.lower().startswith(("rtsp://", "http://", "https://"))

    def _open_stream_capture(self):
        if self._is_remote_stream():
            capture = cv2.VideoCapture(str(self.source), cv2.CAP_FFMPEG)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture
        return cv2.VideoCapture(self.source)

    def _run_stream(self) -> None:
        is_remote_stream = self._is_remote_stream()
        capture = self._open_stream_capture()
        if not capture.isOpened():
            if is_remote_stream:
                self.error_occurred.emit(
                    "无法打开远程视频流。请确认地址是连续视频流地址，例如 rtsp://... 或 http://.../video，"
                    "不要填写监控网页地址或单张截图地址。"
                )
            else:
                self.error_occurred.emit("无法打开视频或摄像头。")
            return

        writer = None
        total_frames = 0
        if self.mode == DetectionMode.VIDEO and not is_remote_stream:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 0 or total_frames > 1_000_000_000:
                total_frames = 0

        stream_message = ""
        try:
            if self.mode == DetectionMode.VIDEO and self.temp_video_path and self.write_video_result:
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(capture.get(cv2.CAP_PROP_FPS)) or self.display_fps
                writer = cv2.VideoWriter(
                    self.temp_video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

            frame_index = 0
            reconnect_count = 0
            max_reconnects = 8 if is_remote_stream else 0
            while self._running:
                if not self._wait_if_paused():
                    break

                ok, frame = capture.read()
                if not ok:
                    if is_remote_stream and reconnect_count < max_reconnects:
                        reconnect_count += 1
                        self.status_changed.emit(f"远程视频流中断，正在重连({reconnect_count}/{max_reconnects})...")
                        capture.release()
                        self.msleep(900)
                        capture = self._open_stream_capture()
                        if capture.isOpened():
                            continue
                        self.msleep(1200)
                        continue
                    if is_remote_stream and frame_index == 0:
                        stream_message = (
                            "远程视频流没有读到有效画面。常见原因：地址是网页/截图地址、手机或摄像头推流已停止、"
                            "电脑与摄像头不在同一网络，或账号密码/IP/端口不正确。"
                        )
                    elif is_remote_stream:
                        stream_message = "远程视频流已断开，检测已停止。"
                    break

                reconnect_count = 0
                frame_index += 1
                source_name = "camera" if self.mode == DetectionMode.CAMERA else Path(str(self.source)).name
                emit_payload = frame_index == 1 or frame_index % self.stream_emit_interval == 0
                need_annotated = (
                    emit_payload
                    or writer is not None
                    or (self.mode == DetectionMode.CAMERA and self._camera_recording)
                )
                annotated = self._process_frame(
                    frame,
                    source_name=source_name,
                    frame_index=frame_index,
                    progress_current=frame_index,
                    progress_total=total_frames,
                    emit_payload=emit_payload,
                    need_annotated=need_annotated,
                )
                if writer is not None and annotated is not None:
                    writer.write(annotated)

                self.msleep(1)
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if self._camera_writer is not None:
                self._camera_writer.release()
                self._camera_writer = None

        if self._running:
            if is_remote_stream and stream_message:
                if "没有读到有效画面" in stream_message:
                    self.error_occurred.emit(stream_message)
                else:
                    self.detection_finished.emit(stream_message)
            elif is_remote_stream:
                self.detection_finished.emit("远程视频流检测已停止")
            else:
                self.detection_finished.emit("视频检测完成" if self.mode == DetectionMode.VIDEO else "摄像头检测已停止")

    def run(self) -> None:
        try:
            if self.mode == DetectionMode.FOLDER:
                self._run_folder()
            elif self.mode in {DetectionMode.VIDEO, DetectionMode.CAMERA}:
                self._run_stream()
            else:
                self.error_occurred.emit("当前线程模式不支持。")
        except Exception as exc:
            self.error_occurred.emit(f"检测过程中发生异常: {exc}")
