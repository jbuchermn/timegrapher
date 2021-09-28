from __future__ import annotations
from typing import Optional, Callable, TYPE_CHECKING

import time
import matplotlib.pyplot as plt
from timegrapher import Timegrapher, PATTERN_TICKS

if TYPE_CHECKING:
    from control import Control

HALF_YLIM_MS = 20

class Display:
    def __init__(self, control: Control, timegrapher: Timegrapher):
        super().__init__()
        self._control: Control = control
        self._timegrapher: Timegrapher = timegrapher

    def run(self):
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line1, = ax.plot(list(range(PATTERN_TICKS)), [0] * PATTERN_TICKS, 'r.', alpha=.3)
        line2, = ax.plot(list(range(PATTERN_TICKS)), [0] * PATTERN_TICKS, 'b.')

        def crop(x):
            while x > HALF_YLIM_MS:
                x -= 2*HALF_YLIM_MS
            while x < -HALF_YLIM_MS:
                x += 2*HALF_YLIM_MS
            return x

        while True:
            ax.set_ylim((-HALF_YLIM_MS, HALF_YLIM_MS))

            line1.set_xdata([i for i, x in enumerate(self._timegrapher.pattern) if x is None])
            line1.set_ydata([0 for x in self._timegrapher.pattern if x is None])

            line2.set_xdata([i for i, x in enumerate(self._timegrapher.pattern) if x is not None])
            line2.set_ydata([crop(x) for x in self._timegrapher.pattern if x is not None])

            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(.1)
