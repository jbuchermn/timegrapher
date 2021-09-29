from __future__ import annotations
from typing import Optional, Callable

import time
import matplotlib as mpl
import matplotlib.pyplot as plt

from control import Control
from capture import Capture
from timegrapher import Timegrapher
from display import Display



if __name__ == '__main__':

    control = Control()
    tg = Timegrapher(control)

    Capture(control, "hw:2,0", tg).start()

    print("Starting display...")
    Display(control, tg).run()
