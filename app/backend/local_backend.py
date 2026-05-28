import csv
import queue
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class LocalBackend:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._queue = queue.Queue()
        self._stop_token = object()
        self._writer = threading.Thread(target=self._writer_loop, daemon=True)
        self._init_schema()
        self._writer.start()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;

                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    mode TEXT NOT NULL,
                    source TEXT,
                    model_name TEXT,
                    model_path TEXT,
                    runtime TEXT,
                    imgsz INTEGER,
                    conf REAL,
                    iou REAL,
                    status TEXT DEFAULT 'running',
                    message TEXT
                );

                CREATE TABLE IF NOT EXISTS frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    source_name TEXT,
                    frame_index INTEGER,
                    fps REAL,
                    box_count INTEGER,
                    progress_current INTEGER,
                    progress_total INTEGER,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_id INTEGER NOT NULL,
                    target_index INTEGER,
                    class_id INTEGER,
                    class_name TEXT,
                    confidence REAL,
                    FOREIGN KEY(frame_id) REFERENCES frames(id)
                );
                """
            )
            self.conn.commit()

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._stop_token:
                    return
                session_id, payload = item
                self._write_frame(session_id, payload)
            finally:
                self._queue.task_done()

    def start_session(
        self,
        *,
        mode: str,
        source: str,
        model_name: str,
        model_path: str,
        runtime: str,
        imgsz: int,
        conf: float,
        iou: float,
    ) -> int:
        with self._lock:
            cursor = self.conn.execute(
                """
                INSERT INTO sessions (
                    started_at, mode, source, model_name, model_path, runtime, imgsz, conf, iou
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    mode,
                    source,
                    model_name,
                    model_path,
                    runtime,
                    imgsz,
                    conf,
                    iou,
                ),
            )
            self.conn.commit()
            return int(cursor.lastrowid)

    def log_frame(self, session_id: int, payload: Dict) -> None:
        self._write_frame(session_id, payload)

    def log_frame_async(self, session_id: int, payload: Dict) -> None:
        payload_copy = {
            "source_name": payload.get("source_name", ""),
            "frame_index": payload.get("frame_index", 0),
            "fps": payload.get("fps", 0.0),
            "box_count": payload.get("box_count", 0),
            "progress_current": payload.get("progress_current", 0),
            "progress_total": payload.get("progress_total", 0),
            "rows": [dict(row) for row in payload.get("rows", [])],
        }
        self._queue.put((session_id, payload_copy))

    def _write_frame(self, session_id: int, payload: Dict) -> None:
        with self._lock:
            cursor = self.conn.execute(
                """
                INSERT INTO frames (
                    session_id, created_at, source_name, frame_index, fps, box_count,
                    progress_current, progress_total
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    datetime.now().isoformat(timespec="seconds"),
                    payload.get("source_name", ""),
                    int(payload.get("frame_index", 0) or 0),
                    float(payload.get("fps", 0.0) or 0.0),
                    int(payload.get("box_count", 0) or 0),
                    int(payload.get("progress_current", 0) or 0),
                    int(payload.get("progress_total", 0) or 0),
                ),
            )
            frame_id = int(cursor.lastrowid)
            rows = payload.get("rows", []) or []
            self.conn.executemany(
                """
                INSERT INTO detections (
                    frame_id, target_index, class_id, class_name, confidence
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        frame_id,
                        int(row.get("index", 0) or 0),
                        int(row.get("class_id", -1) or -1),
                        str(row.get("class_name", "")),
                        float(row.get("confidence", 0.0) or 0.0),
                    )
                    for row in rows
                ],
            )
            self.conn.commit()

    def finish_session(self, session_id: Optional[int], *, status: str, message: str = "") -> None:
        if session_id is None:
            return
        self.flush()
        with self._lock:
            self.conn.execute(
                """
                UPDATE sessions
                SET finished_at = ?, status = ?, message = ?
                WHERE id = ?
                """,
                (datetime.now().isoformat(timespec="seconds"), status, message, session_id),
            )
            self.conn.commit()

    def export_detections_csv(self, session_id: int, output_path: str) -> int:
        self.flush()
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT
                    s.id AS session_id,
                    s.mode,
                    s.source,
                    s.model_name,
                    s.runtime,
                    s.imgsz,
                    f.source_name,
                    f.frame_index,
                    f.fps,
                    f.box_count,
                    d.target_index,
                    d.class_id,
                    d.class_name,
                    d.confidence
                FROM sessions s
                JOIN frames f ON f.session_id = s.id
                LEFT JOIN detections d ON d.frame_id = f.id
                WHERE s.id = ?
                ORDER BY f.frame_index, d.target_index
                """,
                (session_id,),
            ).fetchall()

        fieldnames = [
            "session_id",
            "mode",
            "source",
            "model_name",
            "runtime",
            "imgsz",
            "source_name",
            "frame_index",
            "fps",
            "box_count",
            "target_index",
            "class_id",
            "class_name",
            "confidence",
        ]
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        return len(rows)

    def list_sessions(self, limit: int = 80):
        self.flush()
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT
                    s.id,
                    s.started_at,
                    s.finished_at,
                    s.mode,
                    s.source,
                    s.model_name,
                    s.runtime,
                    s.imgsz,
                    s.conf,
                    s.iou,
                    s.status,
                    s.message,
                    COUNT(DISTINCT f.id) AS frame_count,
                    COUNT(d.id) AS detection_count,
                    AVG(f.fps) AS avg_fps,
                    MAX(f.fps) AS max_fps
                FROM sessions s
                LEFT JOIN frames f ON f.session_id = s.id
                LEFT JOIN detections d ON d.frame_id = f.id
                GROUP BY s.id
                ORDER BY s.id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def search_results(self, keyword: str = "", limit: int = 300):
        self.flush()
        pattern = f"%{keyword.strip()}%" if keyword.strip() else "%"
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT
                    s.id AS session_id,
                    s.started_at,
                    s.mode,
                    s.source,
                    s.model_name,
                    s.runtime,
                    s.status,
                    f.source_name,
                    f.frame_index,
                    f.fps,
                    f.box_count,
                    d.target_index,
                    d.class_id,
                    d.class_name,
                    d.confidence
                FROM sessions s
                LEFT JOIN frames f ON f.session_id = s.id
                LEFT JOIN detections d ON d.frame_id = f.id
                WHERE
                    s.source LIKE ?
                    OR s.model_name LIKE ?
                    OR s.runtime LIKE ?
                    OR s.status LIKE ?
                    OR f.source_name LIKE ?
                    OR d.class_name LIKE ?
                ORDER BY s.id DESC, f.frame_index DESC, d.target_index
                LIMIT ?
                """,
                (pattern, pattern, pattern, pattern, pattern, pattern, int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def session_summary(self, session_id: int):
        self.flush()
        with self._lock:
            session = self.conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (int(session_id),),
            ).fetchone()
            if session is None:
                return {}

            aggregate = self.conn.execute(
                """
                SELECT
                    COUNT(DISTINCT f.id) AS frame_count,
                    COUNT(d.id) AS detection_count,
                    AVG(f.fps) AS avg_fps,
                    MIN(f.fps) AS min_fps,
                    MAX(f.fps) AS max_fps,
                    AVG(d.confidence) AS avg_conf,
                    MIN(d.confidence) AS min_conf,
                    MAX(d.confidence) AS max_conf
                FROM frames f
                LEFT JOIN detections d ON d.frame_id = f.id
                WHERE f.session_id = ?
                """,
                (int(session_id),),
            ).fetchone()
            classes = self.conn.execute(
                """
                SELECT d.class_name, COUNT(*) AS count, AVG(d.confidence) AS avg_conf
                FROM detections d
                JOIN frames f ON f.id = d.frame_id
                WHERE f.session_id = ?
                GROUP BY d.class_name
                ORDER BY count DESC, d.class_name
                """,
                (int(session_id),),
            ).fetchall()

        return {
            "session": dict(session),
            "aggregate": dict(aggregate) if aggregate is not None else {},
            "classes": [dict(row) for row in classes],
        }

    def flush(self) -> None:
        self._queue.join()

    def close(self) -> None:
        self.flush()
        self._queue.put(self._stop_token)
        self._queue.join()
        self._writer.join(timeout=2)
        with self._lock:
            self.conn.close()
