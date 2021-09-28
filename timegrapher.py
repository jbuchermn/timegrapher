from __future__ import annotations
from typing import Optional, Callable, Union, TYPE_CHECKING

import math
import numpy as np

if TYPE_CHECKING:
    from capture import Tick
    from control import Control

PATTERN_TICKS = 100

class Timegrapher:
    def __init__(self, control: Control):
        self._control: Control = control

        self._at_tock: bool = False

        self._last_tick: Optional[Tick] = None
        self._last_tock: Optional[Tick] = None

        self._pattern_ref_timestamp: Optional[float] = None
        self._pattern_idx: int = 0
        self.pattern: list[Optional[float]] = [None] * PATTERN_TICKS

        self.rate: list[tuple[float, Optional[float]]] = []
        self.beat_error: list[tuple[float, Optional[float]]] = []
        self.amplitude: list[tuple[float, Optional[float], Optional[float]]] = []

        self.tick_wave: tuple[np.ndarray, np.ndarray] = (np.zeros(shape=(1,)), np.zeros(shape=(1,)))
        self.tock_wave: tuple[np.ndarray, np.ndarray] = (np.zeros(shape=(1,)), np.zeros(shape=(1,)))


    def reset(self):
        self._pattern = []
        self._rate = []
        self._beat_error = []
        self._amplitude = []


    def __call__(self, tick: Union[Tick, float]) -> None:
        rate: Optional[float] = None
        beat_error: Optional[float] = None
        dt: Optional[float] = None

        if not isinstance(tick, float):
            dt = tick.get_final_timestamp() - tick.get_start_timestamp()

        if self._at_tock:
            if not isinstance(tick, float) and self._last_tick is not None:
                rate = tick.get_start_timestamp() - self._last_tick.get_start_timestamp()
            if not isinstance(tick, float) and self._last_tick is not None and self._last_tock is not None:
                beat_error = np.abs(tick.get_start_timestamp() - 2*self._last_tick.get_start_timestamp() +
                                    self._last_tock.get_start_timestamp())

            self._last_tock = None if isinstance(tick, float) else tick
        else:
            if not isinstance(tick, float) and self._last_tock is not None:
                rate = tick.get_start_timestamp() - self._last_tock.get_start_timestamp()
            if not isinstance(tick, float) and self._last_tick is not None and self._last_tock is not None:
                beat_error = np.abs(tick.get_start_timestamp() - 2*self._last_tock.get_start_timestamp() +
                                    self._last_tick.get_start_timestamp())

            self._last_tick = None if isinstance(tick, float) else tick

        if not isinstance(tick, float):
            self.rate += [(tick.get_start_timestamp(), rate)]
            self.beat_error += [(tick.get_start_timestamp(), beat_error)]
            amplitude: Optional[float] = None
            if dt is not None:
                amplitude = 3600. * self._control.mvmt_lift_angle / (dt * math.pi * self._control.mvmt_bph)

            if self._at_tock:
                self.amplitude[-1] = (self.amplitude[-1][0], self.amplitude[-1][1], amplitude)
                self.tock_wave = tick.get_wave()
            else:
                self.amplitude += [(tick.get_start_timestamp(), amplitude, None)]
                self.tick_wave = tick.get_wave()
        else:
            self.rate += [(tick, None)]
            self.beat_error += [(tick, None)]
            if not self._at_tock:
                self.amplitude += [(tick, None, None)]

        if self._pattern_ref_timestamp is None:
            self._pattern_ref_timestamp = tick.get_start_timestamp() if not isinstance(tick, float) else tick

        pos = None
        if not isinstance(tick, float):
            pos = tick.get_start_timestamp() - self._pattern_idx*self._control.get_mvmt_timescale_ms() - self._pattern_ref_timestamp

        self.pattern[self._pattern_idx] = pos
        self._pattern_idx = (self._pattern_idx + 1) % PATTERN_TICKS
        if self._pattern_idx == 0:
            self._pattern_ref_timestamp += PATTERN_TICKS * self._control.get_mvmt_timescale_ms()

        self._at_tock = not self._at_tock
