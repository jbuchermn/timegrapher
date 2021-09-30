from __future__ import annotations
from typing import Optional, Callable

import sys
import time
from threading import Thread
import matplotlib as mpl
import matplotlib.pyplot as plt


from control import Control
from capture import Capture
from timegrapher import Timegrapher
from display import Display
from multiprocessing.connection import Listener, Client

address = ('localhost', 6000)

if __name__ == '__main__':

    if len(sys.argv) < 2 or sys.argv[1] == "--server":
        print("Starting timegrapher...")
        control = Control()
        tg: Timegrapher = Timegrapher(control)

        Capture(control, "hw:2,0", tg).start()

        listener = Listener(address, authkey='timegrapher'.encode("ascii"))
        while True:
            conn = listener.accept()
            print('Connection accepted from', listener.last_accepted)
            try:
                while True:
                    msg = conn.recv()
                    conn.send(tg)
            except:
                print("Connection closed")

        listener.close()

    else:
        display: Optional[Display] = None

        def thread():
            global display

            while True:
                try:
                    conn = Client(address, authkey='timegrapher'.encode("ascii"))
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
