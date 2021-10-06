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
from multiprocessing.connection import Listener, Client

MP_ADDRESS = ('localhost', 6000)
MP_AUTH = "timegrapher".encode("ascii")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Timegrapher')
    parser.add_argument('-d', '--display', action='store_true')
    parser.add_argument('-b', '--bph', type=int)
    parser.add_argument('-t', '--tick-threshold', type=float)
    parser.add_argument('-p', '--peak-threshold', type=float)
    args = parser.parse_args()

    if not args.display:
        print("Starting timegrapher...")
        control = Control(args.bph, args.tick_threshold, args.peak_threshold)

        tg = Timegrapher(control)
        capture = Capture(control, "hw:2,0", tg)
        listener = Listener(MP_ADDRESS, authkey=MP_AUTH)

        capture.start()

        try:
            while True:
                conn = listener.accept()
                print('Connection accepted from', listener.last_accepted)
                try:
                    while True:
                        msg = conn.recv()
                        conn.send(tg)
                except:
                    print("Connection closed")
        finally:
            listener.close()
            capture.stop()

    else:
        display: Optional[Display] = None

        def thread():
            global display

            while True:
                try:
                    conn = Client(MP_ADDRESS, authkey=MP_AUTH)
                    print("Connection established")
                    while True:
                        conn.send('tick')
                        tg = conn.recv()
                        if display is None:
                            display = Display(tg)
                        else:
                            display.update_timegrapher(tg)
                        time.sleep(.1)
                    conn.close()
                except:
                    print("Connection closed")
                time.sleep(1)

        Thread(target=thread).start()

        while True:
            if display is not None:
                print("Starting display...")
                display.run()
