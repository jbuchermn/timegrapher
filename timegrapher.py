from __future__ import annotations
from typing import Optional, Callable, Union, TYPE_CHECKING

import math
import numpy as np
import time

if TYPE_CHECKING:
    from capture import Tick
    from control import Control

PATTERN_TICKS = 100
TIME_SERIES_AVG_RECENT = 20
TIME_SERIES_SKIP_RECENT = 4
TIME_SERIES_EXTENT_RECENT = 100
TIME_SERIES_AVG_LONG = 100
TIME_SERIES_SKIP_LONG = 20

class TimeSeries:
    def __init__(self):
        # Timestamp, value, stddev
        self.series_recent: list[tuple[float, float, float]] = []
        self.series_long: list[tuple[float, float, float]] = []
        self._current: list[tuple[float, float]] = []
        self._recent_idx: int = 0

    def _calculate(self, n, skip):
        vals = np.array([v for t, v in self._current[len(self._current)-n:]])
        t = 0.5*(self._current[len(self._current)-n][0] + self._current[-1][0])

        temp = np.argpartition(vals, skip)
        skip_low = temp[:skip]
        temp = np.argpartition(-vals, skip)
        skip_high = temp[:skip]
        vals_ma = np.ma.array(vals, mask=False)
        vals_ma.mask[skip_low] = True
        vals_ma.mask[skip_high] = True
        return t, vals_ma.mean(), vals_ma.std()

    def __call__(self, timestamp: float, value: float):
        self._current += [(timestamp, value)]
        if len(self._current) - self._recent_idx >= TIME_SERIES_AVG_RECENT:
            self.series_recent += [self._calculate(TIME_SERIES_AVG_RECENT, TIME_SERIES_SKIP_RECENT)]
            self._recent_idx += TIME_SERIES_AVG_RECENT
            if len(self.series_recent) > TIME_SERIES_EXTENT_RECENT:
                self.series_recent = self.series_recent[1:]

        if len(self._current) >= TIME_SERIES_AVG_LONG:
            self.series_long += [self._calculate(len(self._current), TIME_SERIES_SKIP_LONG)]
            self._current = []
            self._recent_idx = 0

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
                if self._last_tock is not None:
                    rate = (tick.get_start_timestamp() - self._last_tock.get_start_timestamp()) / 2.
                if self._last_tick is not None and self._last_tock is not None:
                    beat_error = np.abs(tick.get_start_timestamp() - 2*self._last_tick.get_start_timestamp() +
                                        self._last_tock.get_start_timestamp())

                self._last_tock = tick
            else:
                if self._last_tick is not None:
                    rate = (tick.get_start_timestamp() - self._last_tick.get_start_timestamp()) / 2.
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
                self.rate(tick.get_start_timestamp(), self.calculate_rate(rate))
            if beat_error is not None:
                self.beat_error(tick.get_start_timestamp(), beat_error)

            amplitude: Optional[float] = None
            if dt is not None:
                amplitude = self.calculate_amplitude(dt)

            ts, vals = tick.get_wave()
            wave = (ts, vals, tick.get_final_timestamp() - tick.get_start_timestamp())
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

    def calculate_rate(self, dt):
        return (24 * 3600) - (dt * self._control.mvmt_bph * 24. / 1000.)

    def calculate_amplitude(self, dt):
        if dt <= 0.:
            return 0
        return min(360, 3600000. * self._control.mvmt_lift_angle / (dt * math.pi * self._control.mvmt_bph))
