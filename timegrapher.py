from __future__ import annotations
from typing import Optional, Callable, Union, TYPE_CHECKING

import math
import numpy as np

if TYPE_CHECKING:
    from capture import Tick
    from control import Control

LIFT_ANGLE = 52
PATTERN_TICKS = 100

class TimeSeries:
    def __init__(self):
        self.ts: list[float] = []
        self.raw: list[float] = []
        self._smooth: float = 0.
        self.smooth: list[float] =[]

    def __call__(self, timestamp: float, value: float):
        self.ts += [timestamp]
        self.raw += [value]

        self._smooth = self._smooth * 0.95 + value * 0.05
        self.smooth += [self._smooth]

class Timegrapher:
    def __init__(self, control: Control):
        self._control: Control = control

        self._at_tock: bool = False

        self._last_tick: Optional[Tick] = None
        self._last_tock: Optional[Tick] = None

        self._pattern_ref_timestamp: Optional[float] = None
        self._pattern_idx: int = 0
        self.pattern: list[Optional[float]] = [None] * PATTERN_TICKS

        self.rate: TimeSeries = TimeSeries()
        self.beat_error: TimeSeries = TimeSeries()
        self.amplitude_tick: TimeSeries = TimeSeries()
        self.amplitude_tock: TimeSeries = TimeSeries()

        self.tick_wave: tuple[np.ndarray, np.ndarray, float] = (np.zeros(shape=(1,)), np.zeros(shape=(1,)), 0)
        self.tock_wave: tuple[np.ndarray, np.ndarray, float] = (np.zeros(shape=(1,)), np.zeros(shape=(1,)), 0)

    def reset(self):
        self._pattern = []
        self._rate = []
        self._beat_error = []
        self._amplitude = []


    def __call__(self, tick: Union[Tick, float]) -> None:
        rate: Optional[float] = None
        beat_error: Optional[float] = None
        dt: Optional[float] = None

        """
        Calculate rate, beat_error, amplitude (dt), Book-keeping
        """
        if not isinstance(tick, float):
            dt = tick.get_final_timestamp() - tick.get_start_timestamp()

            if self._at_tock:
                if self._last_tick is not None:
                    rate = tick.get_start_timestamp() - self._last_tick.get_start_timestamp()
                if self._last_tick is not None and self._last_tock is not None:
                    beat_error = np.abs(tick.get_start_timestamp() - 2*self._last_tick.get_start_timestamp() +
                                        self._last_tock.get_start_timestamp())

                self._last_tock = tick
            else:
                if self._last_tock is not None:
                    rate = tick.get_start_timestamp() - self._last_tock.get_start_timestamp()
                if self._last_tick is not None and self._last_tock is not None:
                    beat_error = np.abs(tick.get_start_timestamp() - 2*self._last_tock.get_start_timestamp() +
                                        self._last_tick.get_start_timestamp())
                self._last_tick = tick
        else:
            if self._at_tock:
                self._last_tock = None
            else:
                self._last_tick = None


        """
        Store rate, beat_error, amplitude and wave-form
        """
        if not isinstance(tick, float):
            if rate is not None:
                rate = (24 * 3600) - (rate * self._control.mvmt_bph * 24. / 1000.)
                self.rate(tick.get_start_timestamp(), rate)
            if beat_error is not None:
                self.beat_error(tick.get_start_timestamp(), beat_error)

            amplitude: Optional[float] = None
            if dt is not None:
                amplitude = 3600000. * LIFT_ANGLE / (dt * math.pi * self._control.mvmt_bph)

            ts, vals = tick.get_wave()
            wave = (ts, vals/max(np.max(vals), -np.min(vals)), tick.get_final_timestamp() - tick.get_start_timestamp())
            if self._at_tock:
                if amplitude is not None:
                    self.amplitude_tock(tick.get_start_timestamp(), amplitude)
                self.tock_wave = wave
            else:
                if amplitude is not None:
                    self.amplitude_tick(tick.get_start_timestamp(), amplitude)
                self.tick_wave = wave

        """
        Pattern
        """
        if self._pattern_ref_timestamp is None:
            self._pattern_ref_timestamp = tick.get_start_timestamp() if not isinstance(tick, float) else tick

        pos = None
        if not isinstance(tick, float):
            pos = tick.get_start_timestamp() - self._pattern_idx*self._control.get_mvmt_timescale_ms() - self._pattern_ref_timestamp

        self.pattern[self._pattern_idx] = pos
        self._pattern_idx = (self._pattern_idx + 1) % PATTERN_TICKS
        if self._pattern_idx == 0:
            self._pattern_ref_timestamp += PATTERN_TICKS * self._control.get_mvmt_timescale_ms()

        """
        Book-keeping
        """
        self._at_tock = not self._at_tock
