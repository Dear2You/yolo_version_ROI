from dataclasses import dataclass


@dataclass
class AppConfig:
    app_name: str = "Visual Interface for YOLO"
    organization: str = "ChenYV"
    app_id: str = "yolo_visual_project"
    window_title: str = ""
    window_width: int = 1240
    window_height: int = 720
    display_fps: int = 30
    stream_ui_emit_interval: int = 5
    folder_ui_emit_interval: int = 10
    auto_write_video_result: bool = False
    save_folder_results: bool = False
    inference_imgsz: int = 416
    resource_poll_interval_ms: int = 10000
    analysis_benchmark_sizes: tuple = (320, 416, 640)
    analysis_benchmark_limit: int = 50
    default_low_confidence_alert: float = 0.35
    default_box_jump_alert: int = 5
    default_no_target_alert_frames: int = 5
    default_class_count_alert: int = 10
    default_roi_percent: tuple = (20, 20, 60, 60)
    default_confidence: float = 0.25
    default_iou: float = 0.45
    record_fps: int = 30
    max_detection_records: int = 20000
    temp_dir_name: str = "yolo_visual_project"
    backend_dir_name: str = "_runtime"
    backend_db_name: str = "detection_backend.sqlite3"
    supported_image_exts: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    supported_video_exts: tuple = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
