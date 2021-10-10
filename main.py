from __future__ import annotations
from typing import Optional, Callable

import time
import argparse
from threading import Thread
import matplotlib as mpl
import matplotlib.pyplot as plt


from control import Control
from capture import Capture
from timegrapher import Timegrapher
from display import Display

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Timegrapher')
    parser.add_argument('-d', '--no-display', action='store_true')
    parser.add_argument('-b', '--bph', type=int)
    parser.add_argument('-t', '--tick-threshold', type=float)
    parser.add_argument('-p', '--peak-threshold', type=float)
    parser.add_argument('-l', '--lift-angle', type=float, default=52)
    args = parser.parse_args()

    print("Starting timegrapher...")
    control = Control(args.lift_angle, args.bph, args.tick_threshold, args.peak_threshold)

    tg = Timegrapher(control)
    capture = Capture(control, "hw:2,0", tg)
    display = None
    if not args.no_display:
        display = Display(control, tg)

    capture.start()

    try:
        if display is not None:
            display.run()
        else:
            while True:
                time.sleep(1)
    finally:
        capture.stop()
