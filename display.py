from __future__ import annotations
from typing import Optional, Callable, TYPE_CHECKING

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from timegrapher import Timegrapher, PATTERN_TICKS

if TYPE_CHECKING:
    from control import Control

HALF_YLIM_MS = 20

class Display:
    def __init__(self, timegrapher: Timegrapher):
        super().__init__()
        self._timegrapher: Timegrapher = timegrapher

    def update_timegrapher(self, timegrapher):
        self._timegrapher = timegrapher

    def run(self):
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(3, 4)

        ax_pattern = fig.add_subplot(gs[:2,:2])
        ax_wave1 = fig.add_subplot(gs[2, 0])
        ax_wave2 = fig.add_subplot(gs[2, 1])
        ax_rate = fig.add_subplot(gs[0, 2:])
        ax_amplitude = fig.add_subplot(gs[1, 2:])
        ax_beat_error = fig.add_subplot(gs[2, 2:])

        ax_pattern.set_title("Timegrapher")
        ax_wave1.set_title("Tick waveform")
        ax_wave2.set_title("Tock waveform")
        ax_rate.set_title("Rate")
        ax_amplitude.set_title("Amplitude")
        ax_beat_error.set_title("Beat error")

        ax_pattern.xaxis.set_ticklabels([])
        ax_pattern.set_ylim((-HALF_YLIM_MS, HALF_YLIM_MS))

        line_pattern1, = ax_pattern.plot(list(range(PATTERN_TICKS)), [0] * PATTERN_TICKS, 'r.', alpha=.3)
        line_pattern2, = ax_pattern.plot(list(range(PATTERN_TICKS)), [0] * PATTERN_TICKS, 'b.')
        line_pattern3, = ax_pattern.plot(list(range(PATTERN_TICKS)), [0] * PATTERN_TICKS, 'g.')

        ax_wave1.yaxis.set_ticklabels([])
        ax_wave2.yaxis.set_ticklabels([])
        ax_wave1.set_xlim((-.2*self._timegrapher.mvmt_timescale_ms, .2*self._timegrapher.mvmt_timescale_ms))
        ax_wave1.set_ylim((-1, 1.))
        ax_wave2.set_xlim((-.2*self._timegrapher.mvmt_timescale_ms, .2*self._timegrapher.mvmt_timescale_ms))
        ax_wave2.set_ylim((-1., 1.))

        line_wave11, = ax_wave1.plot([0, 0], [0, 0], 'b-', linewidth=1)
        line_wave12, = ax_wave1.plot([0, 0], [0, 0], 'b-', linewidth=1)
        line_wave21, = ax_wave2.plot([0, 0], [0, 0], 'b-', linewidth=1)
        line_wave22, = ax_wave2.plot([0, 0], [0, 0], 'b-', linewidth=1)

        ax_amplitude.set_ylim((0, 360))

        ax_rate.xaxis.set_ticklabels([])
        ax_amplitude.xaxis.set_ticklabels([])
        ax_beat_error.xaxis.set_ticklabels([])

        line_rate, = ax_rate.plot([0], [0], 'b.')
        line_amplitude1, = ax_amplitude.plot([0], [0], 'y-')
        line_amplitude2, = ax_amplitude.plot([0], [0], 'g-')
        line_beat_error, = ax_beat_error.plot([0], [0], 'r.')

        def crop(x):
            while x > HALF_YLIM_MS:
                x -= 2*HALF_YLIM_MS
            while x < -HALF_YLIM_MS:
                x += 2*HALF_YLIM_MS
            return x

        def update(_):
            line_pattern1.set_xdata([i for i, x in enumerate(self._timegrapher.pattern) if x is None])
            line_pattern1.set_ydata([0 for x in self._timegrapher.pattern if x is None])

            p2 = [x for i, x in enumerate(self._timegrapher.pattern) if i%2 == 0]
            p3 = [x for i, x in enumerate(self._timegrapher.pattern) if i%2 == 1]
            line_pattern2.set_xdata([2*i for i, x in enumerate(p2) if x is not None])
            line_pattern2.set_ydata([crop(x) for x in p2 if x is not None])
            line_pattern3.set_xdata([2*i+1 for i, x in enumerate(p3) if x is not None])
            line_pattern3.set_ydata([crop(x) for x in p3 if x is not None])

            ts, ys, final = self._timegrapher.tick_wave
            line_wave11.set_xdata(ts)
            line_wave11.set_ydata(ys)
            line_wave12.set_xdata(ts)
            line_wave12.set_ydata(-ys)

            ts, ys, final = self._timegrapher.tock_wave
            line_wave21.set_xdata(ts)
            line_wave21.set_ydata(ys)
            line_wave22.set_xdata(ts)
            line_wave22.set_ydata(-ys)

            if len(self._timegrapher.rate.ts) > 1:
                ys = self._timegrapher.get_rate_smooth()
                ax_rate.set_xlim((min(self._timegrapher.rate.ts), max(self._timegrapher.rate.ts)))
                ax_rate.set_ylim((min(-5, max(-300, min(ys))), max(5, min(300, max(ys)))))
                line_rate.set_xdata(self._timegrapher.rate.ts)
                line_rate.set_ydata(ys)

            if len(self._timegrapher.amplitude_tick.ts) > 1:
                ax_amplitude.set_xlim((min(self._timegrapher.amplitude_tick.ts), max(self._timegrapher.amplitude_tick.ts)))
                line_amplitude1.set_xdata(self._timegrapher.amplitude_tick.ts)
                line_amplitude1.set_ydata(self._timegrapher.amplitude_tick.smooth)

            if len(self._timegrapher.amplitude_tock.ts) > 1:
                line_amplitude2.set_xdata(self._timegrapher.amplitude_tock.ts)
                line_amplitude2.set_ydata(self._timegrapher.amplitude_tock.smooth)

            if len(self._timegrapher.beat_error.ts) > 1:
                ax_beat_error.set_xlim((min(self._timegrapher.beat_error.ts), max(self._timegrapher.beat_error.ts)))
                ax_beat_error.set_ylim((min(self._timegrapher.beat_error.smooth)), max(self._timegrapher.beat_error.smooth))
                line_beat_error.set_xdata(self._timegrapher.beat_error.ts)
                line_beat_error.set_ydata(self._timegrapher.beat_error.smooth)

            return [line_pattern1, line_pattern2, line_wave11, line_wave12, line_wave21, line_wave22, line_rate, line_amplitude1, line_amplitude2, line_beat_error]

        anim = animation.FuncAnimation(fig, update, None, interval=100, blit=False)
        plt.show()
