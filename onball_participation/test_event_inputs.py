"""Checks for the two points at which event labels or oracle conditions could leak."""

import unittest

import numpy as np
import pandas as pd
import torch

from train_events import EventModel, action_profiles, predict, swap_slots


class EventInputs(unittest.TestCase):
    def test_prior_uses_only_earlier_training_days(self):
        match = np.arange(1, 6)
        val = np.array([False, False, False, True, True])
        meta = pd.DataFrame(
            dict(
                match_id=match,
                date=[10, 10, 11, 12, 13],
                block=np.where(val, "model_validation", "model_train"),
            )
        )
        player = np.array([[10, 20]] * 3 + [[10, 99]] * 2)
        group = np.ones_like(player)
        slot = zone = np.zeros(5, int)

        def profiles(labels):
            return action_profiles(meta, match, val, slot, zone, np.array(labels), player, group)

        role, person, group_table, personal_table, key = profiles([0, 11, 10, 7, 12])
        self.assertTrue(np.all(role[:2] == 0) and np.all(person[:2] == 0))
        self.assertTrue((person[2] != 0).any())
        modified = profiles([0, 11, 8, 7, 12])
        np.testing.assert_array_equal(modified[0], role)
        np.testing.assert_array_equal(modified[1], person)
        for a, b in zip(
            profiles([0, 11, 10, 3, 8]), (role, person, group_table, personal_table, key)
        ):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(personal_table[key[-1, 1]], group_table[group[-1, 1]])

    def test_forecast_ignores_oracle_fields(self):
        torch.manual_seed(17)
        n = 4
        model = EventModel(13)
        torch.nn.init.normal_(model.area.weight)
        torch.nn.init.normal_(model.profile.weight)
        data = {
            "token": torch.ones(n, 32, dtype=torch.long),
            "wzone": torch.zeros(n, 32, dtype=torch.long),
            "context": torch.randn(n, 3),
            "age": torch.randn(n, 32),
            "slot": torch.zeros(n, dtype=torch.long),
            "zone": torch.zeros(n, dtype=torch.long),
        }
        q = torch.rand(n, 3, 9)
        q /= q.sum((1, 2), keepdim=True)
        key = torch.arange(3).expand(n, -1)
        lookup = {"personal": torch.randn(3, 9, 13)}
        group = np.tile([0, 1, 0], (n, 1))
        perm = torch.tensor(swap_slots(np.tile([18, 42, 51], (n, 1)), group))
        np.testing.assert_array_equal(perm, np.tile([2, 1, 0], (n, 1)))

        def probabilities():
            return predict(
                model, "personal", data, torch.arange(n), q, lookup, {"personal": key}, perm, True
            )

        before = probabilities()
        data["slot"][:] = 1
        data["zone"][:] = 8
        after = probabilities()
        for field in ("predicted", "swap"):
            torch.testing.assert_close(before[field], after[field])
        self.assertFalse(torch.allclose(before["known_both"], after["known_both"]))


if __name__ == "__main__":
    unittest.main()
