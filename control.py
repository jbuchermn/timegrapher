from __future__ import annotations

from capture import MS_PER_FRAME

class Control:
    def __init__(self):
        # TODO: Detect
        self.mvmt_bph: int = 18000

        self.mvmt_lift_angle: float = 52

        # TODO: Control loop based on number of M / E
        self.tick_threshold: int = 800

        # TODO: Control loop based on number of detected ticks per tick and quality of beat error
        # (missing the correct first tick often means beat error jumps a lot)
        self.peak_threshold: float = 0.01

    def get_mvmt_timescale_ms(self) -> float:
        return 3600000. / self.mvmt_bph

    def get_mvmt_timescale_frames(self) -> float:
        return self.get_mvmt_timescale_ms() / MS_PER_FRAME
