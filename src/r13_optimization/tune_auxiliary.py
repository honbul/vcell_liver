"""Register supplemental families without changing the first-round tuning recipe."""

from . import tune


if __name__ == "__main__":
    tune.FAMILIES += ("dose_transfer",)
    tune.main()
