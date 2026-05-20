"""Progressive bit-depth ladder data pipeline (Lyra 2 proposal §6.6.2)."""
from .posterize import posterize_luma_ladder, rgb_to_luma_709, posterize_luma
