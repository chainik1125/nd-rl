"""Shared figure axes must include sampled outcomes above greedy outcomes."""
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import report


class Figures(unittest.TestCase):
    def test_sampled_maximum_is_visible_and_missing_modes_have_no_legend(self):
        endpoints={key:{"solved":n,"cohorts_above_reference":{"6":{"solved":0},"12":{"solved":0}}}
                   for key,n in {"sft_greedy":2,"rl_greedy":5,"sft_sampled":4,"rl_sampled":7}.items()}
        results={"complete":False,"arms":{"invented":{"cap":10,"mode":"cot","endpoints":endpoints}}}
        figures=[]
        with patch.object(report,"save",side_effect=lambda fig,*_:figures.append(fig)):
            report.coverage_figures(Path("/unused"),results)
        self.addCleanup(lambda:[plt.close(fig) for fig in figures])
        for ax in figures[0].axes:
            self.assertGreater(ax.get_ylim()[1],7)
        labels=figures[0].axes[1].get_legend_handles_labels()[1]
        self.assertEqual(labels,["CoT · SFT","CoT · RL"])


if __name__ == "__main__":
    unittest.main()
