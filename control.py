from __future__ import annotations
from typing import Optional

import numpy as np

from capture import MS_PER_FRAME

BEAT_RATES = [18000, 21600, 28800, 36000]
CONTROL_UPDATE_INTERVAL = 20

class Control:
    def __init__(self, lift_angle: float, fixed_bph: Optional[int] = None, fixed_tick_threshold: Optional[float] = None, fixed_peak_threshold: Optional[float] = None):
        self._fixed = (fixed_bph is not None, fixed_tick_threshold is not None, fixed_peak_threshold is not None)

        self.mvmt_lift_angle: float = lift_angle
        self.mvmt_bph: int = 18000 if fixed_bph is None else fixed_bph
        self.tick_threshold: float = 0.2 if fixed_tick_threshold is None else fixed_tick_threshold
        self.peak_threshold: float = 0.01 if fixed_peak_threshold is None else fixed_peak_threshold

        self._current_misses: int = 0
        self._current_errors: int = 0
        self._current_ticks: int = 0
        self._timestamps: list[float] = []

        self._find_br_counter: int = 0

    def get_mvmt_timescale_ms(self) -> float:
        return 3600000. / self.mvmt_bph

    def get_mvmt_timescale_frames(self) -> float:
        return self.get_mvmt_timescale_ms() / MS_PER_FRAME

    def _find_beatrate(self):
        dts = [self._timestamps[i] - self._timestamps[i-1] for i in range(1, len(self._timestamps))]
        if len(dts) < 3:
            return None
        dts = np.sort(dts)
        m = np.mean(dts[1:-1])
        s = np.std(dts[1:-1])
        for br in BEAT_RATES:
            if np.abs(m - 3600000. / br) < 0.1*m and s < 0.2*m:
                return br

    def _update(self):
        if self._current_misses + self._current_ticks + self._current_errors < CONTROL_UPDATE_INTERVAL:
            return

        if not self._fixed[0]:
            if (self._current_errors + self._current_misses) > self._current_ticks:
                if self._find_br_counter >= 3:
                    print("Looking for different beatrate...")
                    br = self._find_beatrate()
                    if br is None:
                        print("...unsuccessful")
                    elif br != self.mvmt_bph:
                        print("...updating %d -> %d" % (self.mvmt_bph, br))
                        self.mvmt_bph = br
                    else:
                        print("...still fine")
                else:
                    self._find_br_counter += 1
            else:
                self._find_br_counter = 0


        if not self._fixed[1]:
            if self._current_errors > 2:
                self.tick_threshold *= 1.1
                print("Increasing tick threshold %.3f..." % self.tick_threshold)
            elif self._current_misses > self._current_errors + 2:
                self.tick_threshold /= 1.1
                print("Reducing tick threshold %.3f..." % self.tick_threshold)

        # TODO: Peak threshold control loop based on number of detected ticks per tick and quality of beat error
        # (missing the correct first tick often means beat error jumps a lot)

        self._current_misses = 0
        self._current_errors = 0
        self._current_ticks = 0
        self._timestamps = []

    def miss(self, timestamp: float):
        self._current_misses += 1
        self._update()

    def error(self, timestamp):
        self._current_errors += 1
        self._timestamps += [timestamp]
        self._update()

    def tick(self, timestamp: float):
        self._current_ticks += 1
        self._timestamps += [timestamp]
        self._update()

    def no_signal(self):
        self.tick_threshold /= 2
