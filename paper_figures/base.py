"""
base.py — Abstract base class for all paper figure generators.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class FigureGenerator(ABC):
    """
    Single Responsibility: generate one publication-quality figure.

    Subclasses implement `generate()` to build and return a matplotlib Figure.
    The `save()` method is provided by this base class and handles PDF export.
    """

    @abstractmethod
    def generate(self) -> Figure:
        """Build and return the matplotlib Figure. Must not call plt.show()."""
        ...

    def save(self, path: Path) -> None:
        """
        Generate the figure and save it as a vector PDF.

        Args:
            path: destination file path (parent directory must exist).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = self.generate()
        fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"  [saved] {path.name}  ({path.stat().st_size // 1024} KB)")
