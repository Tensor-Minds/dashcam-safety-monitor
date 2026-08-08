"""Configurable temporal rule evaluation and highest-priority alert fusion."""

from __future__ import annotations

import os
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from ml_pipeline.rule_config import RuleConfiguration


ANOMALY_LABELS = {
    "accident": "Accident",
    "road accident": "Accident",
    "road_accident": "Accident",
    "car fire": "Car Fire",
    "car-fire": "Car Fire",
    "car_fire": "Car Fire",
    "vehicle fire": "Car Fire",
    "vehicle_fire": "Car Fire",
    "fighting": "Fighting",
    "snatching": "Snatching",
}
SPEED_LIMIT = re.compile(r"^speed[ _-]*limit[ _-]*(\d{1,3})$", re.IGNORECASE)
ROAD_SIGN_MESSAGES = {
    "Green-Light": "Green traffic light detected.",
    "Red-Light": "Warning. Red traffic light detected ahead.",
    "Stop": "Warning. Stop sign detected ahead.",
    "Stop-Ahead": "Warning. Stop sign detected ahead.",
    "Bus-Stop": "Bus stop detected.",
    "Children-Present-Or-Crossing-Ahead": "Warning. Children crossing detected ahead. Slow down and be prepared to stop.",
    "Children-Crossing": "Warning. Children crossing detected ahead. Slow down and be prepared to stop.",
    "Double-Bend-To-Left-Ahead": "Warning. Double bend to left ahead.",
    "Double-Bend-To-Right-Ahead": "Warning. Double bend to right ahead.",
    "Left-Bend-Ahead": "Warning. Left bend ahead.",
    "Narrow-Bridge-Or-Culvert-Ahead": "Warning. Narrow bridge detected ahead.",
    "Pedestrian-Crossing": "Warning. Pedestrian crossing detected ahead. Slow down and be prepared to stop.",
    "Pedestrian-Crossing-Ahead": "Warning. Pedestrian crossing detected ahead. Slow down and be prepared to stop.",
    "Right-Bend-Ahead": "Warning. Right bend ahead.",
    "T-Junction-Ahead": "Warning. T junction ahead.",
    "Traffic-From-Left-Merges-Ahead": "Warning. Traffic merging from left ahead.",
    "Traffic-From-Right-Merges-Ahead": "Warning. Traffic merging from right ahead.",
    "Level-Crossing-With-Gates": "Warning. Level crossing detected ahead.",
    "Hospital": "Hospital zone ahead.",
    "No-Honking": "No honking zone.",
    "No-Left-Turn": "No left turn permitted.",
    "No-Right-Turn": "No right turn permitted.",
    "No-U-Turn": "No U-turn permitted.",
}


def calculate_iou(left: List[int], right: List[int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / (left_area + right_area - intersection)


@dataclass
class SpatialTrack:
    track_id: str
    category: str
    class_name: str
    bbox: List[int]
    first_seen_ms: float
    last_seen_ms: float
    history: Deque[bool] = field(default_factory=lambda: deque(maxlen=30))
    confidence_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=30)
    )
    missed: int = 0
    last_rule_alert_ms: Dict[str, float] = field(default_factory=dict)


class ReportAlertPipeline:
    """Evaluate every configured rule, then communicate the highest priority."""

    def __init__(self, rule_config: Optional[RuleConfiguration] = None) -> None:
        self.rule_config = rule_config or RuleConfiguration()
        self.tracks: Dict[str, SpatialTrack] = {}
        self.next_track_id = 1
        self.last_audio_ms = -1e12
        self.last_communication_ms = -1e12
        self.last_primary: Optional[Dict[str, Any]] = None
        self.smoothed_lane_offset: Optional[float] = None
        self.lane_departure_started_ms: Optional[float] = None
        self.lane_direction: Optional[str] = None
        self.last_quality_status_ms = -1e12
        self.blur_threshold = self._configured_blur_threshold()

    def _configured_blur_threshold(self) -> float:
        return float(self.rule_config.system["minimum_blur_score"])

    def update_rule_config(self, rule_config: RuleConfiguration) -> None:
        self.rule_config = rule_config
        self.blur_threshold = self._configured_blur_threshold()
        self.reset()

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1
        self.last_audio_ms = -1e12
        self.last_communication_ms = -1e12
        self.last_primary = None
        self.smoothed_lane_offset = None
        self.lane_departure_started_ms = None
        self.lane_direction = None
        self.last_quality_status_ms = -1e12

    def evaluate(
        self,
        frame: np.ndarray,
        raw_detections: List[Dict[str, Any]],
        timestamp_ms: float,
        turn_signal: str = "off",
        vehicle_speed_kmh: Optional[float] = None,
        is_video: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
        height, width = frame.shape[:2]
        quality = self._image_quality(frame, timestamp_ms)
        detections = self._filter_detections(raw_detections, width, height)
        detections = self._annotate_spatial_context(detections, width, height)
        if not is_video:
            primary = self._single_frame_primary(
                detections, timestamp_ms, vehicle_speed_kmh
            )
            return detections, primary, quality

        tracked = self._update_tracks(detections, timestamp_ms)
        candidates: List[Dict[str, Any]] = []
        if quality["usable"]:
            candidates.extend(
                self._tracked_candidates(tracked, timestamp_ms, vehicle_speed_kmh)
            )
            lane_alert = self._lane_candidate(
                detections, width, height, timestamp_ms, turn_signal
            )
            if lane_alert:
                candidates.append(lane_alert)
        else:
            self.lane_departure_started_ms = None

        primary = self._fuse(candidates, timestamp_ms) if quality["usable"] else None
        for detection in detections:
            detection["alert_state"] = "raw"
            if primary and detection.get("track_id") == primary.get("track_id"):
                detection["alert_state"] = "active"
        return detections, primary, quality

    def _image_quality(
        self, frame: np.ndarray, timestamp_ms: float
    ) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_brightness = float(gray.mean())
        minimum_brightness = float(
            self.rule_config.system["minimum_mean_brightness"]
        )
        maximum_brightness = float(
            self.rule_config.system["maximum_mean_brightness"]
        )
        usable = (
            blur_score >= self.blur_threshold
            and minimum_brightness <= mean_brightness <= maximum_brightness
        )
        status_trigger = bool(
            not usable
            and timestamp_ms - self.last_quality_status_ms
            >= self.rule_config.system["image_quality_status_cooldown_ms"]
        )
        if status_trigger:
            self.last_quality_status_ms = timestamp_ms
        return {
            "blur_score": round(blur_score, 2),
            "minimum_blur_score": self.blur_threshold,
            "mean_brightness": round(mean_brightness, 2),
            "minimum_mean_brightness": minimum_brightness,
            "maximum_mean_brightness": maximum_brightness,
            "usable": usable,
            "status_trigger": status_trigger,
            "message": (
                "Low camera visibility - alerts are temporarily suppressed"
                if status_trigger
                else None
            ),
        }

    def _filter_detections(
        self, detections: List[Dict[str, Any]], width: int, height: int
    ) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        sign_filter = self.rule_config.data["road_sign_filter"]
        pothole_filter = self.rule_config.data["pothole_filter"]
        anomaly_filter = self.rule_config.data["anomaly_filter"]
        for original in detections:
            detection = dict(original)
            category = detection.get("category")
            confidence = float(detection.get("confidence", 0))
            box = detection.get("bbox")
            if not box or len(box) != 4:
                continue
            if category == "road_sign":
                if confidence < sign_filter["minimum_confidence"]:
                    continue
                box_width = max(1, box[2] - box[0])
                box_height = max(1, box[3] - box[1])
                aspect_ratio = box_width / box_height
                area_ratio = box_width * box_height / max(1, width * height)
                if (
                    not sign_filter["minimum_aspect_ratio"]
                    <= aspect_ratio
                    <= sign_filter["maximum_aspect_ratio"]
                    or area_ratio < sign_filter["minimum_frame_area_ratio"]
                ):
                    continue
            elif (
                category == "pothole"
                and confidence < pothole_filter["minimum_confidence"]
            ):
                continue
            elif category == "anomaly":
                if confidence < anomaly_filter["minimum_confidence"]:
                    continue
                key = str(detection.get("class_name", "")).strip().lower()
                if key not in ANOMALY_LABELS:
                    continue
                detection["class_name"] = ANOMALY_LABELS[key]
            elif (
                category == "lane_line"
                and confidence < self.rule_config.data["lane"]["minimum_confidence"]
            ):
                continue
            rule_label, _ = self._rule_label_and_values(detection, None)
            detection["priority"] = max(
                (
                    rule["priority"]
                    for rule in self.rule_config.matching_rules(
                        str(category), rule_label
                    )
                ),
                default=0,
            )
            filtered.append(detection)
        return filtered

    def _annotate_spatial_context(
        self, detections: List[Dict[str, Any]], width: int, height: int
    ) -> List[Dict[str, Any]]:
        """Pass through detections directly without spatial proximity filtering."""
        return detections

    def _update_tracks(
        self, detections: List[Dict[str, Any]], timestamp_ms: float
    ) -> List[Tuple[SpatialTrack, Dict[str, Any]]]:
        trackable = [
            detection
            for detection in detections
            if detection.get("category") in {"pothole", "anomaly", "road_sign"}
            or (
                detection.get("category") == "lane_line"
                and str(detection.get("class_name", "")).strip().lower()
                in {"crossing", "crosswalk", "pedestrian crossing", "pedestrian-crossing"}
            )
        ]
        unmatched = set(self.tracks)
        updated: List[Tuple[SpatialTrack, Dict[str, Any]]] = []
        for detection in sorted(
            trackable, key=lambda item: float(item.get("confidence", 0)), reverse=True
        ):
            match = self._best_track(detection, unmatched)
            if match is None:
                track_id = f"{detection['category']}-{self.next_track_id}"
                self.next_track_id += 1
                match = SpatialTrack(
                    track_id=track_id,
                    category=detection["category"],
                    class_name=str(detection.get("class_name", "")),
                    bbox=detection["bbox"],
                    first_seen_ms=timestamp_ms,
                    last_seen_ms=timestamp_ms,
                )
                self.tracks[track_id] = match
            else:
                unmatched.discard(match.track_id)
            match.bbox = detection["bbox"]
            match.last_seen_ms = timestamp_ms
            match.missed = 0
            match.history.append(True)
            match.confidence_history.append(float(detection.get("confidence", 0)))
            detection["track_id"] = match.track_id
            updated.append((match, detection))
        for track_id in unmatched:
            track = self.tracks[track_id]
            track.history.append(False)
            track.missed += 1
        for track_id in [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed > self.rule_config.system["maximum_track_misses"]
        ]:
            del self.tracks[track_id]
        return updated

    def _best_track(
        self, detection: Dict[str, Any], available: Set[str]
    ) -> Optional[SpatialTrack]:
        candidates = [
            track
            for track_id, track in self.tracks.items()
            if track_id in available
            and track.category == detection.get("category")
            and track.class_name.lower()
            == str(detection.get("class_name", "")).lower()
        ]
        candidates.sort(
            key=lambda track: calculate_iou(track.bbox, detection["bbox"]), reverse=True
        )
        threshold = self.rule_config.system["tracking_iou_threshold"]
        if (
            candidates
            and calculate_iou(candidates[0].bbox, detection["bbox"]) > threshold
        ):
            return candidates[0]
        return None

    def _tracked_candidates(
        self,
        tracked: List[Tuple[SpatialTrack, Dict[str, Any]]],
        timestamp_ms: float,
        vehicle_speed_kmh: Optional[float],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for track, detection in tracked:
            label, values = self._rule_label_and_values(
                detection, vehicle_speed_kmh
            )
            for rule in self.rule_config.matching_rules(track.category, label):
                if float(detection.get("confidence", 0)) < rule["minimum_confidence"]:
                    continue
                temporal = rule["temporal"]
                hits = (
                    sum(list(track.history)[-temporal["window_frames"] :])
                    if temporal and temporal.get("window_frames")
                    else 1
                )
                duration_ms = timestamp_ms - track.first_seen_ms
                if (
                    temporal
                    and temporal.get("window_frames")
                    and hits < temporal["minimum_hits"]
                ):
                    continue
                if temporal and duration_ms < temporal.get("minimum_duration_ms", 0):
                    continue
                if not self._conditions_match(rule, values):
                    continue
                last_alert = track.last_rule_alert_ms.get(rule["id"], -1e12)
                if timestamp_ms - last_alert < rule["cooldown_ms"]:
                    continue
                candidate_item = self._candidate_from_rule(
                    rule,
                    detection,
                    track.track_id,
                    timestamp_ms,
                    values,
                    {
                        "temporal_hits": hits,
                        "temporal_window": (
                            temporal.get("window_frames") if temporal else None
                        ),
                        "required_hits": (
                            temporal.get("minimum_hits") if temporal else None
                        ),
                        "duration_ms": duration_ms,
                        "required_duration_ms": (
                            temporal.get("minimum_duration_ms", 0)
                            if temporal
                            else 0
                        ),
                        "maximum_misses": (
                            temporal.get("maximum_misses")
                            if temporal and "maximum_misses" in temporal
                            else None
                        ),
                        "spatial_iou_threshold": self.rule_config.system[
                            "tracking_iou_threshold"
                        ],
                    },
                )
                candidates.append(candidate_item)
        return candidates

    def _lane_candidate(
        self,
        detections: List[Dict[str, Any]],
        width: int,
        height: int,
        timestamp_ms: float,
        turn_signal: str,
    ) -> Optional[Dict[str, Any]]:
        lane_items = [
            detection
            for detection in detections
            if detection.get("category") == "lane_line"
            and str(detection.get("class_name", "")).lower()
            in {"line 1", "line 2", "yellow markings"}
        ]
        positions = []
        for detection in lane_items:
            polygon = detection.get("polygon")
            if polygon and len(polygon) >= 3:
                y_values = np.asarray([point[1] for point in polygon], dtype=float)
                x_values = np.asarray([point[0] for point in polygon], dtype=float)
                try:
                    coefficients = np.polyfit(y_values, x_values, 2)
                    positions.append(
                        (float(np.polyval(coefficients, height * 0.9)), detection)
                    )
                except (ValueError, np.linalg.LinAlgError):
                    continue
            else:
                box = detection["bbox"]
                positions.append(((box[0] + box[2]) / 2, detection))
        positions.sort(key=lambda item: item[0])
        if len(positions) < 2:
            self.lane_departure_started_ms = None
            return None
        left_x, left_detection = positions[0]
        right_x, right_detection = positions[-1]
        lane_width = right_x - left_x
        if lane_width <= 1:
            return None
        lane_config = self.rule_config.data["lane"]
        lane_center = (left_x + right_x) / 2
        offset_ratio = (width / 2 - lane_center) / lane_width
        alpha = lane_config["ema_alpha"]
        self.smoothed_lane_offset = (
            offset_ratio
            if self.smoothed_lane_offset is None
            else alpha * offset_ratio + (1 - alpha) * self.smoothed_lane_offset
        )
        if abs(self.smoothed_lane_offset) < lane_config["clear_offset_ratio"]:
            self.lane_departure_started_ms = None
            self.lane_direction = None
            return None
        if abs(self.smoothed_lane_offset) < lane_config["trigger_offset_ratio"]:
            return None
        direction = "left" if self.smoothed_lane_offset < 0 else "right"
        if self.lane_direction != direction:
            self.lane_direction = direction
            self.lane_departure_started_ms = timestamp_ms
            return None
        if self.lane_departure_started_ms is None:
            self.lane_departure_started_ms = timestamp_ms
            return None
        duration = timestamp_ms - self.lane_departure_started_ms
        matching = self.rule_config.matching_rules(
            "lane_line", "Lane Departure"
        )
        if not matching:
            return None
        rule = max(matching, key=lambda item: item["priority"])
        signal_suppresses = rule["conditions"].get(
            "matching_turn_signal_suppresses", False
        )
        unknown_signal_suppresses = rule["conditions"].get(
            "unknown_turn_signal_suppresses", False
        )
        lane_confidence = min(
            float(left_detection.get("confidence", 0)),
            float(right_detection.get("confidence", 0)),
        )
        required_duration_ms = (
            rule["temporal"].get(
                "minimum_duration_ms", lane_config["minimum_duration_ms"]
            )
            if rule["temporal"]
            else lane_config["minimum_duration_ms"]
        )
        if (
            duration < required_duration_ms
            or (signal_suppresses and turn_signal == direction)
            or (
                unknown_signal_suppresses
                and turn_signal not in {"off", "left", "right"}
            )
            or lane_confidence < rule["minimum_confidence"]
        ):
            return None
        return self._candidate_from_rule(
            rule,
            {
                "category": "lane_line",
                "class_name": "Lane Departure",
                "bbox": None,
            },
            f"lane-{direction}",
            timestamp_ms,
            {"label": "Lane Departure", "direction": direction},
            {
                "ema_alpha": alpha,
                "lane_fit_polynomial_order": 2,
                "offset_ratio": round(self.smoothed_lane_offset, 4),
                "trigger_ratio": lane_config["trigger_offset_ratio"],
                "duration_ms": duration,
                "required_duration_ms": required_duration_ms,
                "turn_signal": turn_signal,
                "lane_confidence": lane_confidence,
            },
        )

    def _fuse(
        self,
        candidates: List[Dict[str, Any]],
        timestamp_ms: float,
        persist_state: bool = True,
    ) -> Optional[Dict[str, Any]]:
        higher_priority_interrupt = False
        if (
            persist_state
            and self.last_primary
            and candidates
        ):
            highest_incoming = max(
                candidates,
                key=lambda alert: (
                    alert["priority"],
                    alert.get("confidence", 0),
                ),
            )
            higher_priority_interrupt = bool(
                highest_incoming["priority"] > self.last_primary["priority"]
            )
        if not candidates:
            if persist_state and self.last_primary and self.last_primary["visible_until_ms"] >= timestamp_ms:
                retained = dict(self.last_primary)
                retained["audio_trigger"] = False
                return retained
            return None
        candidates.sort(
            key=lambda alert: (
                alert["priority"],
                alert.get("confidence", 0),
            ),
            reverse=True,
        )
        primary = candidates[0]
        duplicate_threshold = self.rule_config.system["duplicate_iou_threshold"]
        for lower in candidates[1:]:
            if (
                primary.get("bbox")
                and lower.get("bbox")
                and calculate_iou(primary["bbox"], lower["bbox"])
                > duplicate_threshold
            ):
                lower["suppressed_reason"] = "higher_priority_overlapping_alert"
            else:
                lower["suppressed_reason"] = "lower_numeric_priority"
            lower["suppressed_by_rule_id"] = primary["rule_id"]
        rule_cooldown = primary.get("cooldown_ms", 5000)
        primary["audio_trigger"] = bool(primary.get("audio_key")) and (
            not persist_state
            or timestamp_ms - self.last_audio_ms >= rule_cooldown
            or higher_priority_interrupt
        )
        if persist_state and primary["audio_trigger"]:
            self.last_audio_ms = timestamp_ms
        primary["suppressed_alerts"] = candidates[1:]
        if persist_state:
            self.last_communication_ms = timestamp_ms
            self.last_primary = dict(primary)
            if primary.get("track_id") and primary["track_id"] in self.tracks:
                self.tracks[primary["track_id"]].last_rule_alert_ms[primary["rule_id"]] = timestamp_ms
        return primary

    def _single_frame_primary(
        self,
        detections: List[Dict[str, Any]],
        timestamp_ms: float,
        vehicle_speed_kmh: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for detection in detections:
            label, values = self._rule_label_and_values(
                detection, vehicle_speed_kmh
            )
            for rule in self.rule_config.matching_rules(
                detection["category"], label
            ):
                if (
                    float(detection.get("confidence", 0))
                    >= rule["minimum_confidence"]
                    and self._conditions_match(rule, values)
                ):
                    candidates.append(
                        self._candidate_from_rule(
                            rule,
                            detection,
                            None,
                            timestamp_ms,
                            values,
                            {"single_image_mode": True},
                        )
                    )
        # Uploaded images are independent analyses. Video-only cooldown and
        # visible-alert retention must not leak a previous image's alert into
        # the current response.
        return self._fuse(candidates, timestamp_ms, persist_state=False)

    def _rule_label_and_values(
        self,
        detection: Dict[str, Any],
        vehicle_speed_kmh: Optional[float],
    ) -> Tuple[str, Dict[str, Any]]:
        raw_label = str(detection.get("class_name", "Unknown"))
        values: Dict[str, Any] = {
            "label": raw_label.replace("-", " "),
            "vehicle_speed_kmh": vehicle_speed_kmh,
        }
        if detection.get("category") == "road_sign":
            speed_match = SPEED_LIMIT.fullmatch(raw_label)
            if speed_match:
                limit_val = int(speed_match.group(1))
                values["speed_limit_kmh"] = limit_val
                values["sign_message"] = (
                    f"Speed limit {limit_val} kilometers per hour detected."
                )
                return "Speed Limit", values
            canonical = "-".join(
                raw_label.replace("_", " ").replace("-", " ").split()
            ).title()
            sign_msg = ROAD_SIGN_MESSAGES.get(canonical) or ROAD_SIGN_MESSAGES.get(raw_label)
            if not sign_msg:
                sign_msg = f"{values['label']} ahead"
            values["sign_message"] = sign_msg
            return canonical, values
        if detection.get("category") == "pothole":
            return (
                "Pothole" if raw_label.strip().lower() == "pothole" else raw_label,
                values,
            )
        if detection.get("category") == "lane_line":
            clean_lane = raw_label.strip().lower()
            if clean_lane in {"crossing", "crosswalk", "pedestrian crossing", "pedestrian-crossing"}:
                values["label"] = "Pedestrian Crossing"
                return "Crossing", values
        return raw_label, values

    def _conditions_match(
        self, rule: Dict[str, Any], values: Dict[str, Any]
    ) -> bool:
        conditions = rule["conditions"]
        if conditions.get("speed_limit_exceeded"):
            speed = values.get("vehicle_speed_kmh")
            limit = values.get("speed_limit_kmh")
            if speed is None or limit is None:
                return False
            return speed > limit + conditions.get("speed_tolerance_kmh", 0)
        return True

    def _candidate_from_rule(
        self,
        rule: Dict[str, Any],
        detection: Dict[str, Any],
        track_id: Optional[str],
        timestamp_ms: float,
        values: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        safe_values = {
            "label": values.get("label", detection.get("class_name", "Hazard")),
            "direction": values.get("direction", "centre"),
            "vehicle_speed_kmh": (
                round(values["vehicle_speed_kmh"])
                if values.get("vehicle_speed_kmh") is not None
                else "unavailable"
            ),
            "speed_limit_kmh": values.get("speed_limit_kmh", "unavailable"),
            "sign_message": values.get(
                "sign_message", f"{values.get('label', 'Road sign')} ahead"
            ),
        }
        alert_event_id = f"evt-{rule['id']}-{track_id or 'single'}-{int(timestamp_ms)}"
        return {
            "alert_event_id": alert_event_id,
            "rule_id": rule["id"],
            "category": detection["category"],
            "class_name": detection.get("class_name", safe_values["label"]),
            "message": rule["message"].format(**safe_values),
            "bbox": detection.get("bbox"),
            "track_id": track_id,
            "priority": rule["priority"],
            "confidence": float(detection.get("confidence", 0)),
            "audio_key": rule["audio_key"],
            "cooldown_ms": rule.get("cooldown_ms", 5000),
            "severity": rule["severity"],
            "timestamp_ms": timestamp_ms,
            "visible_until_ms": timestamp_ms
            + rule["visual_duration_ms"],
            "evidence": {
                **evidence,
                "rule_id": rule["id"],
                "priority": rule["priority"],
                "priority_policy": self.rule_config.data["priority_policy"],
                "vehicle_speed_kmh": values.get("vehicle_speed_kmh"),
                "vehicle_speed_source": (
                    "simulated"
                    if values.get("vehicle_speed_kmh") is not None
                    else "unavailable"
                ),
                "speed_limit_kmh": values.get("speed_limit_kmh"),
            },
        }
