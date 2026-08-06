"""Validation and persistence for the editable alert rule set."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml


ALLOWED_MODULES = {"anomaly", "lane_line", "pothole", "road_sign"}
ALLOWED_SEVERITIES = {"info", "warning", "high", "critical"}
ALLOWED_AUDIO_KEYS = {None, "anomaly", "lane_departure", "pothole", "road_sign"}
ALLOWED_CONDITIONS = {
    "inside_ego_lane",
    "relative_proximity",
    "speed_limit_exceeded",
    "speed_tolerance_kmh",
    "matching_turn_signal_suppresses",
    "unknown_turn_signal_suppresses",
}
ALLOWED_TEMPORAL_FIELDS = {
    "window_frames",
    "minimum_hits",
    "minimum_duration_ms",
    "maximum_misses",
}


class RuleConfiguration:
    def __init__(self, path: Path | None = None) -> None:
        configured_path = os.getenv("ALERT_RULES_PATH")
        self.path = path or (
            Path(configured_path)
            if configured_path
            else Path(__file__).with_name("rules.yml")
        )
        self.data = self._load()

    @property
    def system(self) -> Dict[str, Any]:
        return self.data["system"]

    @property
    def rules(self) -> List[Dict[str, Any]]:
        return self.data["rules"]

    def matching_rules(self, module: str, label: str) -> List[Dict[str, Any]]:
        return [
            rule
            for rule in self.rules
            if rule["enabled"]
            and rule["module"] == module
            and ("*" in rule["labels"] or label in rule["labels"])
        ]

    def public_data(self) -> Dict[str, Any]:
        return deepcopy(self.data)

    def replace(self, data: Dict[str, Any]) -> None:
        validated = self._validate(data)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            yaml.safe_dump(validated, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
        self.data = validated

    def reload(self) -> None:
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        with self.path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError("Rule configuration must contain a YAML object")
        return self._validate(data)

    def _validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        required_top = {
            "version",
            "priority_policy",
            "system",
            "road_sign_filter",
            "pothole_filter",
            "anomaly_filter",
            "lane",
            "rules",
        }
        unknown = set(data) - required_top
        missing = required_top - set(data)
        if unknown or missing:
            raise ValueError(
                f"Invalid rule configuration fields; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if data["priority_policy"] != "higher_number_wins":
            raise ValueError("Only priority_policy='higher_number_wins' is supported")
        if not isinstance(data["rules"], list) or not data["rules"]:
            raise ValueError("rules must be a non-empty list")
        expected_sections = {
            "system": {
                "processing_target_fps",
                "tracking_iou_threshold",
                "duplicate_iou_threshold",
                "visual_duration_ms",
                "maximum_track_misses",
                "minimum_blur_score",
                "minimum_mean_brightness",
                "maximum_mean_brightness",
                "image_quality_status_cooldown_ms",
            },
            "road_sign_filter": {
                "minimum_confidence",
                "nms_iou_threshold",
                "minimum_aspect_ratio",
                "maximum_aspect_ratio",
                "minimum_frame_area_ratio",
            },
            "pothole_filter": {
                "minimum_confidence",
                "nms_iou_threshold",
                "medium_bottom_y_ratio",
                "near_bottom_y_ratio",
                "fallback_roi_left_ratio",
                "fallback_roi_right_ratio",
                "fallback_roi_minimum_bottom_y_ratio",
            },
            "anomaly_filter": {"minimum_confidence", "nms_iou_threshold"},
            "lane": {
                "minimum_confidence",
                "ema_alpha",
                "trigger_offset_ratio",
                "clear_offset_ratio",
                "minimum_duration_ms",
            },
        }
        for section, expected in expected_sections.items():
            if not isinstance(data[section], dict) or set(data[section]) != expected:
                raise ValueError(f"Section {section} has missing or unknown fields")
            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in data[section].values()
            ):
                raise ValueError(f"Section {section} values must be numeric")
        system = data["system"]
        if not (
            0 < system["tracking_iou_threshold"] <= 1
            and 0 < system["duplicate_iou_threshold"] <= 1
            and 0
            <= system["minimum_mean_brightness"]
            < system["maximum_mean_brightness"]
            <= 255
        ):
            raise ValueError("System IoU or image-quality thresholds are invalid")
        pothole_filter = data["pothole_filter"]
        if not (
            0
            <= pothole_filter["fallback_roi_left_ratio"]
            < pothole_filter["fallback_roi_right_ratio"]
            <= 1
            and 0
            <= pothole_filter["fallback_roi_minimum_bottom_y_ratio"]
            <= pothole_filter["medium_bottom_y_ratio"]
            < pothole_filter["near_bottom_y_ratio"]
            <= 1
        ):
            raise ValueError("Pothole proximity or fallback ROI thresholds are invalid")
        lane = data["lane"]
        if not (
            0 < lane["ema_alpha"] <= 1
            and 0 <= lane["clear_offset_ratio"] < lane["trigger_offset_ratio"]
            and lane["minimum_duration_ms"] >= 0
        ):
            raise ValueError("Lane smoothing, hysteresis, or duration is invalid")
        identifiers = set()
        required_rule = {
            "id",
            "enabled",
            "module",
            "labels",
            "minimum_confidence",
            "severity",
            "priority",
            "temporal",
            "visual_duration_ms",
            "cooldown_ms",
            "message",
            "audio_key",
            "conditions",
        }
        for rule in data["rules"]:
            if not isinstance(rule, dict):
                raise ValueError("Every rule must be a YAML object")
            if set(rule) != required_rule:
                raise ValueError(f"Rule {rule.get('id', '<unknown>')} has invalid fields")
            if not isinstance(rule["id"], str) or not rule["id"].strip():
                raise ValueError("Every rule requires a non-empty string id")
            if not isinstance(rule["enabled"], bool):
                raise ValueError(f"Rule {rule['id']} enabled must be boolean")
            if rule["id"] in identifiers:
                raise ValueError(f"Duplicate rule id: {rule['id']}")
            identifiers.add(rule["id"])
            if rule["module"] not in ALLOWED_MODULES:
                raise ValueError(f"Unsupported module in rule {rule['id']}")
            if (
                not isinstance(rule["minimum_confidence"], (int, float))
                or not 0 <= rule["minimum_confidence"] <= 1
            ):
                raise ValueError(
                    f"Rule {rule['id']} minimum_confidence must be between 0 and 1"
                )
            if rule["severity"] not in ALLOWED_SEVERITIES:
                raise ValueError(f"Rule {rule['id']} has invalid severity")
            if (
                not isinstance(rule["priority"], int)
                or rule["priority"] < 0
                or rule["priority"] > len(data["rules"])
            ):
                raise ValueError(
                    f"Rule {rule['id']} priority must be an integer from 0 to {len(data['rules'])}"
                )
            if not isinstance(rule["labels"], list) or not rule["labels"]:
                raise ValueError(f"Rule {rule['id']} requires at least one label")
            if not all(
                isinstance(label, str) and label.strip() for label in rule["labels"]
            ):
                raise ValueError(f"Rule {rule['id']} labels must be non-empty strings")
            if rule["audio_key"] not in ALLOWED_AUDIO_KEYS:
                raise ValueError(f"Rule {rule['id']} has unsupported audio_key")
            if not isinstance(rule["message"], str) or not rule["message"].strip():
                raise ValueError(f"Rule {rule['id']} requires a message")
            if not isinstance(rule["cooldown_ms"], int) or rule["cooldown_ms"] < 0:
                raise ValueError(f"Rule {rule['id']} has invalid cooldown_ms")
            if (
                not isinstance(rule["visual_duration_ms"], int)
                or rule["visual_duration_ms"] < 0
            ):
                raise ValueError(f"Rule {rule['id']} has invalid visual_duration_ms")
            if not isinstance(rule["conditions"], dict):
                raise ValueError(f"Rule {rule['id']} conditions must be a YAML object")
            invalid_conditions = set(rule["conditions"]) - ALLOWED_CONDITIONS
            if invalid_conditions:
                raise ValueError(
                    f"Rule {rule['id']} has unsupported conditions: {sorted(invalid_conditions)}"
                )
            conditions = rule["conditions"]
            for field in (
                "inside_ego_lane",
                "speed_limit_exceeded",
                "matching_turn_signal_suppresses",
                "unknown_turn_signal_suppresses",
            ):
                if field in conditions and not isinstance(conditions[field], bool):
                    raise ValueError(
                        f"Rule {rule['id']} condition {field} must be boolean"
                    )
            if "speed_tolerance_kmh" in conditions and (
                not isinstance(conditions["speed_tolerance_kmh"], (int, float))
                or conditions["speed_tolerance_kmh"] < 0
            ):
                raise ValueError(
                    f"Rule {rule['id']} speed_tolerance_kmh must be non-negative"
                )
            proximity = conditions.get("relative_proximity")
            if proximity is not None:
                values = proximity if isinstance(proximity, list) else [proximity]
                if (
                    not values
                    or not all(isinstance(value, str) for value in values)
                    or not set(values).issubset({"far", "medium", "near"})
                ):
                    raise ValueError(
                        f"Rule {rule['id']} relative_proximity must use far, medium, or near"
                    )
            temporal = rule["temporal"]
            if temporal is not None:
                if (
                    not isinstance(temporal, dict)
                    or not set(temporal).issubset(ALLOWED_TEMPORAL_FIELDS)
                    or not temporal
                ):
                    raise ValueError(f"Rule {rule['id']} has invalid temporal fields")
                window = temporal.get("window_frames")
                hits = temporal.get("minimum_hits")
                if (window is None) != (hits is None):
                    raise ValueError(
                        f"Rule {rule['id']} temporal window_frames and minimum_hits must be configured together"
                    )
                if window is not None and (
                    not isinstance(window, int)
                    or not isinstance(hits, int)
                    or hits < 1
                    or hits > window
                ):
                    raise ValueError(f"Rule {rule['id']} has invalid temporal voting values")
                for field in ("minimum_duration_ms", "maximum_misses"):
                    if field in temporal and (
                        not isinstance(temporal[field], int) or temporal[field] < 0
                    ):
                        raise ValueError(
                            f"Rule {rule['id']} has invalid temporal {field}"
                        )
        return data
