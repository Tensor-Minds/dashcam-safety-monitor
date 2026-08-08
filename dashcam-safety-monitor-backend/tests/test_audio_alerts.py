import unittest
import numpy as np
import os
from ml_pipeline.rule_config import RuleConfiguration
from ml_pipeline.alert_pipeline import ReportAlertPipeline

class TestAudioAlertsPipeline(unittest.TestCase):

    def setUp(self):
        self.rule_config = RuleConfiguration()
        self.pipeline = ReportAlertPipeline(self.rule_config)
        np.random.seed(42)
        self.dummy_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    def tearDown(self):
        # Restore original rules if modified during tests
        self.rule_config.reload()

    def test_anomaly_classes_audio_alerts(self):
        """Test audio alert messages for all anomaly classes."""
        anomaly_test_cases = [
            ("Accident", "Warning. Possible road accident detected ahead. Slow down safely and keep your distance."),
            ("Car Fire", "Warning. Possible vehicle fire detected ahead. Keep a safe distance."),
            ("Fighting", "Possible dangerous incident detected ahead. Proceed carefully."),
            ("Snatching", "Possible dangerous incident detected ahead. Proceed carefully."),
        ]

        for class_name, expected_message in anomaly_test_cases:
            self.pipeline.reset()
            audio_alerts = []
            for frame_idx in range(6):
                raw_dets = [{
                    "bbox": [100, 100, 300, 300],
                    "confidence": 0.90,
                    "class_name": class_name,
                    "category": "anomaly"
                }]
                _, primary, _ = self.pipeline.evaluate(
                    self.dummy_frame, raw_dets, timestamp_ms=frame_idx * 500, is_video=True
                )
                if primary and primary.get("audio_trigger"):
                    audio_alerts.append(primary)
            
            self.assertTrue(len(audio_alerts) > 0, f"Primary audio alert failed for anomaly: {class_name}")
            primary = audio_alerts[0]
            self.assertEqual(primary["audio_key"], "anomaly")
            self.assertEqual(primary["message"], expected_message)
            self.assertTrue(primary["audio_trigger"])
            self.assertIn("alert_event_id", primary)
            self.assertEqual(primary["cooldown_ms"], 1000)

    def test_road_surface_classes_audio_alerts(self):
        """Test audio alert messages for all road surface/pothole classes."""
        surface_test_cases = [
            ("Pothole", "pothole", "Warning. Possible pothole ahead. Reduce speed safely."),
            ("alligator_crack", "pothole", "Warning. Road-surface damage detected ahead. Proceed carefully."),
            ("longitudinal_crack", "pothole", "Warning. Road-surface damage detected ahead. Proceed carefully."),
            ("transverse_crack", "pothole", "Warning. Road-surface damage detected ahead. Proceed carefully."),
        ]

        for class_name, category, expected_message in surface_test_cases:
            self.pipeline.reset()
            audio_alerts = []
            for frame_idx in range(3):
                raw_dets = [{
                    "bbox": [150, 200, 250, 300],
                    "confidence": 0.85,
                    "class_name": class_name,
                    "category": category
                }]
                _, primary, _ = self.pipeline.evaluate(
                    self.dummy_frame, raw_dets, timestamp_ms=frame_idx * 100, is_video=True
                )
                if primary and primary.get("audio_trigger"):
                    audio_alerts.append(primary)
            
            self.assertTrue(len(audio_alerts) > 0, f"Primary audio alert failed for road surface class: {class_name}")
            primary = audio_alerts[0]
            self.assertEqual(primary["audio_key"], "pothole")
            self.assertEqual(primary["message"], expected_message)

    def test_lane_line_pedestrian_crossing_alert(self):
        """Test lane_line model Crossing / Crosswalk mapped to pedestrian crossing audio warning."""
        crossing_labels = ["Crossing", "Crosswalk", "Pedestrian Crossing", "Pedestrian-Crossing"]

        for label in crossing_labels:
            self.pipeline.reset()
            audio_alerts = []
            for frame_idx in range(3):
                raw_dets = [{
                    "bbox": [400, 500, 600, 600],
                    "confidence": 0.80,
                    "class_name": label,
                    "category": "lane_line"
                }]
                _, primary, _ = self.pipeline.evaluate(
                    self.dummy_frame, raw_dets, timestamp_ms=frame_idx * 100, is_video=True
                )
                if primary and primary.get("audio_trigger"):
                    audio_alerts.append(primary)
            
            self.assertTrue(len(audio_alerts) > 0, f"Primary audio alert failed for lane crossing label: {label}")
            primary = audio_alerts[0]
            self.assertEqual(primary["message"], "Warning. Pedestrian crossing detected ahead. Slow down and be prepared to stop.")
            self.assertEqual(primary["audio_key"], "road_sign")

    def test_ordinary_lane_lines_suppressed_from_spoken_audio(self):
        """Verify ordinary lane lines (Line 1, Line 2, Yellow Markings) do not trigger spoken primary audio."""
        self.pipeline.reset()
        raw_dets = [
            {"bbox": [100, 400, 150, 700], "confidence": 0.90, "class_name": "Line 1", "category": "lane_line"},
            {"bbox": [500, 400, 550, 700], "confidence": 0.90, "class_name": "Line 2", "category": "lane_line"},
            {"bbox": [300, 400, 350, 700], "confidence": 0.90, "class_name": "Yellow Markings", "category": "lane_line"},
        ]
        _, primary, _ = self.pipeline.evaluate(
            self.dummy_frame, raw_dets, timestamp_ms=100.0, turn_signal="off", is_video=True
        )
        self.assertIsNone(primary)

    def test_lane_departure_warning(self):
        """Test confirmed lane departure warning."""
        self.pipeline.reset()
        primary = None
        for i in range(20):
            t_ms = i * 100.0
            raw_dets = [
                {"bbox": [900, 400, 920, 700], "confidence": 0.95, "class_name": "Line 1", "category": "lane_line"},
                {"bbox": [1100, 400, 1120, 700], "confidence": 0.95, "class_name": "Line 2", "category": "lane_line"},
            ]
            _, primary, _ = self.pipeline.evaluate(
                self.dummy_frame, raw_dets, timestamp_ms=t_ms, turn_signal="off", is_video=True
            )

        self.assertIsNotNone(primary)
        self.assertEqual(primary["rule_id"], "lane-departure")
        self.assertIn("Lane departure warning", primary["message"])
        self.assertEqual(primary["audio_key"], "lane_departure")

    def test_road_sign_classes_audio_alerts(self):
        """Test all 35 road sign classes and speed limit exceeded rules."""
        sign_test_cases = [
            ("Green Light", "Green traffic light detected."),
            ("Red Light", "Warning. Red traffic light detected ahead."),
            ("Stop", "Warning. Stop sign detected ahead."),
            ("Stop-Ahead", "Warning. Stop sign detected ahead."),
            ("Bus-Stop", "Bus stop detected."),
            ("children crossing", "Warning. Children crossing detected ahead. Slow down and be prepared to stop."),
            ("Children-Present-or-Crossing-Ahead", "Warning. Children crossing detected ahead. Slow down and be prepared to stop."),
            ("Left-Bend-Ahead", "Warning. Left bend ahead."),
            ("Right-Bend-Ahead", "Warning. Right bend ahead."),
            ("Double-Bend-to-Left-Ahead", "Warning. Double bend to left ahead."),
            ("Double-Bend-to-Right-Ahead", "Warning. Double bend to right ahead."),
            ("Narrow-Bridge-or-Culvert-Ahead", "Warning. Narrow bridge detected ahead."),
            ("T-Junction-Ahead", "Warning. T junction ahead."),
            ("Traffic-From-Left-Merges-Ahead", "Warning. Traffic merging from left ahead."),
            ("Traffic-From-Right-Merges-Ahead", "Warning. Traffic merging from right ahead."),
            ("level crossing with gates", "Warning. Level crossing detected ahead."),
            ("hospital", "Hospital zone ahead."),
            ("no honking", "No honking zone."),
            ("no left turn", "No left turn permitted."),
            ("no right turn", "No right turn permitted."),
            ("no u turn", "No U-turn permitted."),
            ("Speed Limit 60", "Speed limit 60 kilometers per hour detected."),
        ]

        for sign_class, expected_message in sign_test_cases:
            self.pipeline.reset()
            audio_alerts = []
            for frame_idx in range(5):
                raw_dets = [{
                    "bbox": [500, 100, 600, 200],
                    "confidence": 0.85,
                    "class_name": sign_class,
                    "category": "road_sign"
                }]
                _, primary, _ = self.pipeline.evaluate(
                    self.dummy_frame, raw_dets, timestamp_ms=frame_idx * 100, is_video=True
                )
                if primary and primary.get("audio_trigger"):
                    audio_alerts.append(primary)
            
            self.assertTrue(len(audio_alerts) > 0, f"Primary audio alert failed for sign class: {sign_class}")
            primary = audio_alerts[0]
            self.assertEqual(primary["audio_key"], "road_sign")
            self.assertEqual(primary["message"], expected_message)

    def test_speed_limit_exceeded_warning(self):
        """Test speed limit exceeded rule."""
        self.pipeline.reset()
        audio_alerts = []
        for frame_idx in range(6):
            raw_dets = [{
                "bbox": [500, 100, 600, 200],
                "confidence": 0.85,
                "class_name": "Speed Limit 50",
                "category": "road_sign"
            }]
            _, primary, _ = self.pipeline.evaluate(
                self.dummy_frame, raw_dets, timestamp_ms=frame_idx * 500, vehicle_speed_kmh=65.0, is_video=True
            )
            if primary and primary.get("audio_trigger"):
                audio_alerts.append(primary)

        exceeded_alerts = [a for a in audio_alerts if a["rule_id"] == "speed-limit-exceeded"]
        self.assertTrue(len(exceeded_alerts) > 0)
        primary = exceeded_alerts[0]
        self.assertEqual(primary["rule_id"], "speed-limit-exceeded")
        self.assertEqual(primary["message"], "Reduce speed. Current speed is 65 kilometers per hour in a 50 zone.")

    def test_cooldown_and_track_deduplication(self):
        """Test rule cooldown prevents repeat audio trigger for the same track."""
        self.pipeline.reset()
        primary_alerts = []
        for frame_idx in range(6):
            raw_dets = [{
                "bbox": [100, 100, 300, 300],
                "confidence": 0.90,
                "class_name": "Accident",
                "category": "anomaly"
            }]
            _, primary, _ = self.pipeline.evaluate(
                self.dummy_frame, raw_dets, timestamp_ms=frame_idx * 100, is_video=True
            )
            if primary:
                primary_alerts.append(primary)

        first_audio_triggers = [p for p in primary_alerts if p["audio_trigger"]]
        self.assertEqual(len(first_audio_triggers), 1)

        # Subsequent frames within 1000ms cooldown should NOT re-trigger audio
        _, subsequent_primary, _ = self.pipeline.evaluate(
            self.dummy_frame, [{
                "bbox": [100, 100, 300, 300],
                "confidence": 0.90,
                "class_name": "Accident",
                "category": "anomaly"
            }], timestamp_ms=500.0, is_video=True
        )
        self.assertIsNotNone(subsequent_primary)
        self.assertFalse(subsequent_primary["audio_trigger"])

    def test_critical_anomaly_priority_interruption(self):
        """Test critical anomaly alert (priority 14) interrupts lower priority road sign alert (priority 8)."""
        self.pipeline.reset()
        # Step 1: Trigger Red Light sign alert at t=1000ms
        audio_alerts = []
        for frame_idx in range(5):
            raw_dets = [{
                "bbox": [500, 100, 600, 200],
                "confidence": 0.85,
                "class_name": "Red Light",
                "category": "road_sign"
            }]
            _, primary, _ = self.pipeline.evaluate(
                self.dummy_frame, raw_dets, timestamp_ms=1000 + frame_idx * 100, is_video=True
            )
            if primary and primary.get("audio_trigger"):
                audio_alerts.append(primary)

        self.assertTrue(len(audio_alerts) > 0)
        self.assertEqual(audio_alerts[0]["rule_id"], "red-light")

        # Step 2: Now at t=1600ms, critical Accident anomaly occurs (higher priority 14)
        anomaly_audio_alerts = []
        for frame_idx in range(6):
            raw_dets = [
                {"bbox": [500, 100, 600, 200], "confidence": 0.85, "class_name": "Red Light", "category": "road_sign"},
                {"bbox": [100, 100, 300, 300], "confidence": 0.90, "class_name": "Accident", "category": "anomaly"}
            ]
            _, primary, _ = self.pipeline.evaluate(
                self.dummy_frame, raw_dets, timestamp_ms=1600 + frame_idx * 100, is_video=True
            )
            if primary and primary.get("audio_trigger"):
                anomaly_audio_alerts.append(primary)

        self.assertTrue(len(anomaly_audio_alerts) > 0)
        self.assertEqual(anomaly_audio_alerts[0]["rule_id"], "anomaly-accident")

    def test_disabled_rules_suppression(self):
        """Test that disabled rules do not trigger alerts."""
        config_data = self.rule_config.public_data()
        for r in config_data["rules"]:
            if r["id"] == "anomaly-accident":
                r["enabled"] = False
        
        # Test disabled rule in memory without modifying global rules.yml
        rule_cfg = RuleConfiguration()
        rule_cfg.data = config_data
        pipeline = ReportAlertPipeline(rule_cfg)

        primary = None
        for frame_idx in range(6):
            raw_dets = [{
                "bbox": [100, 100, 300, 300],
                "confidence": 0.90,
                "class_name": "Accident",
                "category": "anomaly"
            }]
            _, primary, _ = pipeline.evaluate(
                self.dummy_frame, raw_dets, timestamp_ms=frame_idx * 100, is_video=True
            )

        self.assertIsNone(primary)

    def test_single_image_processing_mode(self):
        """Test single image upload processing mode (is_video=False)."""
        self.pipeline.reset()
        raw_dets = [{
            "bbox": [100, 100, 300, 300],
            "confidence": 0.90,
            "class_name": "Accident",
            "category": "anomaly"
        }]
        _, primary, _ = self.pipeline.evaluate(
            self.dummy_frame, raw_dets, timestamp_ms=0.0, is_video=False
        )
        self.assertIsNotNone(primary)
        self.assertEqual(primary["rule_id"], "anomaly-accident")
        self.assertTrue(primary["audio_trigger"])
        self.assertIn("alert_event_id", primary)

if __name__ == "__main__":
    unittest.main()
