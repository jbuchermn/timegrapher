from typing import Optional

import time
import alsaaudio
import numpy as np
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import librosa

SAMPLE_RATE = 44100.
MS_PER_FRAME = 1000. / SAMPLE_RATE

inp = alsaaudio.PCM(
    alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK,
    device="hw:2,0",
    channels=1,
    rate=int(SAMPLE_RATE),
    format=alsaaudio.PCM_FORMAT_S16_LE,
    periodsize=160,
)

MVMT_TICK_THRESHOLD = 1000
MVMT_BPH = 18000
MVMT_TIMESCALE_MS = 3600000. / MVMT_BPH


HIGHPASS_FREQ = 3000
HIGHPASS_ORDER = 10

highpass = signal.butter(HIGHPASS_ORDER, HIGHPASS_FREQ, 'hp', fs=SAMPLE_RATE, output='sos')

TICK_PRERECORD = int(MVMT_TIMESCALE_MS * 0.15 / MS_PER_FRAME)
TICK_MINLENGTH = int(MVMT_TIMESCALE_MS * 0.3 / MS_PER_FRAME)
TICK_MAXLENGTH = int(MVMT_TIMESCALE_MS * 0.5 / MS_PER_FRAME)
TICK_PEAK_THRESHOLD = 0.01

class Tick:
    def __init__(self):
        self._started: bool = False
        self._buffer: np.ndarray = np.zeros(shape=(TICK_MAXLENGTH,), dtype=np.int16)
        self._buffer_at: int = 0

        self._timestamp: float = 0.
        self._ticks: list[tuple[int, float]] = [] # index of tick, score of tick

    def possibly_append(self, next_input: np.ndarray, timestamp_ms: float):
        """
        Handle next_input PCM audio data and either
        - Append to self if self has started, return True
        - Append or skip if self has finished, return False
        - Skip if self has not started, return True
        """
        max_val = max(-next_input.min(), next_input.max())
        if max_val > MVMT_TICK_THRESHOLD or (self._started and self._buffer_at < TICK_MINLENGTH):
            # Recording

            self._started = True

            l = min(next_input.shape[0], TICK_MAXLENGTH-self._buffer_at)
            self._buffer[self._buffer_at:(self._buffer_at+l)] = next_input[:l]
            self._buffer_at += l
            self._timestamp = timestamp_ms - self._buffer_at * MS_PER_FRAME
            return self._buffer_at < TICK_MAXLENGTH
        else:
            if not self._started:
                # Prerecording

                if self._buffer_at + next_input.shape[0] < TICK_PRERECORD:
                    self._buffer[self._buffer_at:(self._buffer_at+next_input.shape[0])] = next_input
                    self._buffer_at += next_input.shape[0]
                else:
                    l = TICK_PRERECORD - (self._buffer_at + next_input.shape[0] - TICK_PRERECORD)
                    self._buffer[:l] = self._buffer[(TICK_PRERECORD-l):TICK_PRERECORD]
                    self._buffer[(TICK_PRERECORD-next_input.shape[0]):TICK_PRERECORD] = next_input
                    self._buffer_at = TICK_PRERECORD

                return True
            else:
                # Finished

                return False

    def get_wave(self):
        return np.arange(self._buffer.shape[0]) * MS_PER_FRAME, self._buffer

    def plot(self, line: mpl.lines.Line2D):
        if len(self._ticks) == 0:
            self._calculate_ticks()

        for l in list(line.axes.lines):
            if l != line:
                l.remove()

        start = self._ticks[0][0]*MS_PER_FRAME
        t, y = self.get_wave()
        line.axes.set_xlim((start - 0.1*MVMT_TIMESCALE_MS, start + 0.5*MVMT_TIMESCALE_MS))
        lim = max(1000, max(-y.min(), y.max()))
        line.axes.set_ylim((-lim, lim))
        line.set_xdata(t)
        line.set_ydata(y)

        line.axes.axvline(x=start, color='r')
        for t, s in self._ticks[1:]:
            line.axes.axvline(x=t*MS_PER_FRAME, color='g', alpha=s)

        # input("?")



    def _calculate_ticks(self):
        fft_size = 32
        freqs = np.fft.rfftfreq(fft_size, 1./SAMPLE_RATE)

        f_space = np.zeros(shape=(freqs.shape[0], self._buffer.shape[0] // fft_size), dtype=np.float32)
        ts = np.arange(f_space.shape[1]) * MS_PER_FRAME * fft_size
        for i in range(f_space.shape[1]):
            f_space[:, i] = np.real(np.fft.rfft(self._buffer[(i*fft_size):((i+1)*fft_size)]))

        # BEGIN DEBUG
        # print(freqs)
        # print(f_space)
        # fig, ax = plt.subplots(1 + freqs.shape[0], sharex=True)
        # ax[0].plot(*self.get_wave())
        # for f in range(freqs.shape[0]):
        #     ax[f+1].plot(ts, f_space[f])
        #
        # plt.show()
        # input("Done?")
        # END DEBUG

        df_norms = np.zeros(shape=(f_space.shape[1]-1,))
        for i in range(df_norms.shape[0]):
            for f in range(1, freqs.shape[0]):
                if abs(f_space[f, i+1]) < abs(f_space[f, i]):
                    continue
                df_norms[i] += (abs(f_space[f, i+1]) - abs(f_space[f, i]))**2

        df_norms /= np.max(df_norms)

        # Idea
        # 1. Extract all local maxima of df_norms
        # 2. For each maximum there is a corresponding time interval in which the tick occured
        #      - df_norms[i] is difference between f_space[i+1] and f_space[i]
        #      - i.e. df_norms[i] maximum corresponds to tick in interval (i+1): (i+1)*fft_size to (i+2)*fft_size
        # 3. Within that interval pick the maximal absolute t-space value as precise tick location
        # 4. Score tick by absolute value of df_norms maximum

        peaks, _ = signal.find_peaks(df_norms, threshold=TICK_PEAK_THRESHOLD)
        for p in peaks:
            m = np.argmax(np.abs(self._buffer[((p+1)*fft_size):((p+2)*fft_size)]))
            self._ticks += [((p+1)*fft_size + m, df_norms[p])]

        # BEGIN DEBUG
        # fig, ax = plt.subplots(2, sharex=True)
        # ax[0].plot(*self.get_wave())
        # ax[1].plot(ts[1:], df_norms)
        #
        # for t, _ in self._ticks:
        #     ax[0].axvline(x=t*MS_PER_FRAME, color='r')
        #     ax[1].axvline(x=t*MS_PER_FRAME, color='r')
        #
        # plt.show()
        # input("Done?")
        # END DEBUG





plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot([0, 1], [0, 0], 'b-')

timestamp_ms = 0

# plot_counter = 0

tick = Tick()
last_tick = tick
while True:
    l, data = inp.read()
    if l == 0:
        time.sleep(.001)
        continue
    if len(data) == 0:
        continue
    arr = np.frombuffer(data[:(2*l)], dtype=np.int16)
    arr_filtered = signal.sosfilt(highpass, arr)

    timestamp_ms += l * MS_PER_FRAME
    if not tick.possibly_append(arr_filtered, timestamp_ms):
        dt = (tick._timestamp - last_tick._timestamp)
        if dt < 0.9 * MVMT_TIMESCALE_MS:
            print("E")
        elif dt > 1.1 * MVMT_TIMESCALE_MS:
            missed = round(dt/MVMT_TIMESCALE_MS) - 1
            if missed > 0:
                print('M' * missed)

            last_timestamp = last_tick._timestamp + missed*MVMT_TIMESCALE_MS
            dt = tick._timestamp - last_timestamp
            if dt < 0.9 * MVMT_TIMESCALE_MS:
                print("E")
            elif dt > 1.1 * MVMT_TIMESCALE_MS:
                print("E")
            else:
                print("T: %.2f ms" % dt)

            last_tick = tick
        else:
            print("T: %.2f ms" % dt)
            # plot_counter += 1
            # if plot_counter%10 == 0:
            #     tick.plot(line1)
            #     fig.canvas.draw()
            #     fig.canvas.flush_events()

            last_tick = tick

        tick = Tick()

