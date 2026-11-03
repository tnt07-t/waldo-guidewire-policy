"""Invented scores only; no images, features or weights are needed."""

import unittest

from select_preprocessing import confirm_command, search, select_within_one_std


def entry(config, mean, std=0.02):
    return {"config": config, "mean": mean, "std": std}


class SelectionTests(unittest.TestCase):
    def test_cheaper_config_wins_within_one_std(self):
        scored = [entry("small", 0.60), entry("large", 0.61, std=0.03)]
        self.assertEqual(select_within_one_std(scored)["config"], "small")
        scored = [entry("small", 0.55), entry("large", 0.61, std=0.03)]
        self.assertEqual(select_within_one_std(scored)["config"], "large")

    def test_search_order_and_stage_skipping(self):
        means = {256: 0.60, 384: 0.70, 192: 0.50, 512: 0.69}
        seen = []

        def score(config):
            seen.append(dict(config))
            mean = means[config["crop_size"]]
            if config["channels"] == "enhanced":
                mean += 0.1 if config.get("difference") == "median" else 0.0
                mean += 0.05 if config.get("tophat_size") == 7 else 0.0
            return entry(config, mean, std=0.005)

        best, stages = search(score, [512, 192, 256, 384], [11, 7])
        self.assertEqual([stage["parameter"] for stage in stages],
                         ["crop_size", "channels", "difference", "tophat_size"])
        # Crop sizes are tried cheapest first; 384 beats 512 by more than one std.
        self.assertEqual([c["crop_size"] for c in seen[:4]], [192, 256, 384, 512])
        self.assertEqual(best, {"crop_size": 384, "channels": "enhanced",
                                "tophat_size": 7, "difference": "median"})

        def raw_wins(config):
            return entry(config, 0.6 if config["channels"] == "raw" else 0.5)

        best, stages = search(raw_wins, [256], [11, 7])
        self.assertEqual(best, {"crop_size": 256, "channels": "raw"})
        self.assertEqual([stage["parameter"] for stage in stages], ["crop_size", "channels"])

    def test_confirm_command_uses_selected_flags(self):
        command = confirm_command({"crop_size": 256, "channels": "enhanced",
                                   "tophat_size": 11, "difference": "median"}, 0)
        self.assertIn("--preprocessor wire --crop-size 256 --channels enhanced", command)
        self.assertIn("--tophat-size 11 --difference median", command)
        self.assertIn("--encoders resnet18", command)
        self.assertNotIn("--standardize-channels", command)
        standardized = confirm_command({"crop_size": 256, "channels": "enhanced"}, 0, True)
        self.assertIn("--standardize-channels", standardized)
        raw = confirm_command({"crop_size": 256, "channels": "raw"}, 0, True)
        self.assertNotIn("--standardize-channels", raw)


if __name__ == "__main__":
    unittest.main()
