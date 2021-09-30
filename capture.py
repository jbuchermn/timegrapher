from __future__ import annotations
from typing import Optional, Callable, Union,TYPE_CHECKING

from threading import Thread
import time
import alsaaudio
import numpy as np
from scipy import signal

if TYPE_CHECKING:
    from control import Control

SAMPLE_RATE = 44100.
MS_PER_FRAME = 1000. / SAMPLE_RATE

HIGHPASS_FREQ = 3000
HIGHPASS_ORDER = 10

TICK_PRERECORD = 0.2
TICK_MINLENGTH = 0.3
TICK_MAXLENGTH = 0.5

TIMESTAMP_FILTER = 0.15

TICK_PEAK_FFT_SIZE = 64


class Tick:
    def __init__(self, control: Control):
        self._control: Control = control

        self._started: bool = False
        self._buffer: np.ndarray = np.zeros(shape=(int(TICK_MAXLENGTH*self._control.get_mvmt_timescale_frames()),),
                                            dtype=np.int16)
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
        max_val = max(-next_input.min(), next_input.max()) / 2.**15
        if max_val > self._control.tick_threshold or (
            self._started and self._buffer_at < TICK_MINLENGTH*self._control.get_mvmt_timescale_frames()):
            # Recording

            self._started = True

            l = min(next_input.shape[0], int(TICK_MAXLENGTH*self._control.get_mvmt_timescale_frames())-self._buffer_at)
            self._buffer[self._buffer_at:(self._buffer_at+l)] = next_input[:l]
            self._buffer_at += l
            self._timestamp = timestamp_ms - self._buffer_at * MS_PER_FRAME
            return self._buffer_at < int(TICK_MAXLENGTH*self._control.get_mvmt_timescale_frames())
        else:
            if not self._started:
                # Prerecording

                prerecord = int(TICK_PRERECORD * self._control.get_mvmt_timescale_frames())
                if self._buffer_at + next_input.shape[0] < prerecord:
                    self._buffer[self._buffer_at:(self._buffer_at+next_input.shape[0])] = next_input
                    self._buffer_at += next_input.shape[0]
                else:
                    l = prerecord - (self._buffer_at + next_input.shape[0] - prerecord)
                    self._buffer[:l] = self._buffer[(prerecord-l):prerecord]
                    self._buffer[(prerecord-next_input.shape[0]):prerecord] = next_input
                    self._buffer_at = prerecord

                return True
            else:
                # Finished

                return False

    def get_wave(self):
        if len(self._ticks) == 0:
            self._calculate_ticks()

        return (np.arange(self._buffer.shape[0]) - self._ticks[0][0])*MS_PER_FRAME, self._buffer

    def get_start_timestamp(self):
        if len(self._ticks) == 0:
            self._calculate_ticks()

        return self._timestamp + self._ticks[0][0]*MS_PER_FRAME

    def get_final_timestamp(self):
        if len(self._ticks) == 0:
            self._calculate_ticks()

        if len(self._ticks) == 1:
            return self._timestamp + self._ticks[0][0]*MS_PER_FRAME

        i = 1
        v = self._ticks[1][1]
        for j in range(2, len(self._ticks)):
            if self._ticks[j][1] > v:
                i = j
                v = self._ticks[j][1]

        return self._timestamp + self._ticks[i][0]*MS_PER_FRAME


    def _calculate_ticks(self):
        freqs = np.fft.rfftfreq(TICK_PEAK_FFT_SIZE, 1./SAMPLE_RATE)

        f_space = np.zeros(shape=(freqs.shape[0], self._buffer.shape[0] // TICK_PEAK_FFT_SIZE), dtype=np.float32)
        ts = np.arange(f_space.shape[1]) * MS_PER_FRAME * TICK_PEAK_FFT_SIZE
        for i in range(f_space.shape[1]):
            f_space[:, i] = np.real(np.fft.rfft(self._buffer[(i*TICK_PEAK_FFT_SIZE):((i+1)*TICK_PEAK_FFT_SIZE)]))

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
        #      - i.e. df_norms[i] maximum corresponds to tick in interval (i+1): (i+1)*TICK_PEAK_FFT_SIZE to (i+2)*TICK_PEAK_FFT_SIZE
        # 3. Within that interval pick the maximal absolute t-space value as precise tick location
        # 4. Score tick by absolute value of df_norms maximum or by absolute t-space value

        peaks, _ = signal.find_peaks(df_norms, threshold=self._control.peak_threshold)
        for p in peaks:
            m = np.argmax(np.abs(self._buffer[((p+1)*TICK_PEAK_FFT_SIZE):((p+2)*TICK_PEAK_FFT_SIZE)]))
            # score = df_norms[p]
            score = np.abs(self._buffer[(p+1)*TICK_PEAK_FFT_SIZE + m])/2.**15
            self._ticks += [((p+1)*TICK_PEAK_FFT_SIZE + m, score)]

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

        if len(self._ticks) == 0:
            self._ticks = [(0, 0)]


class Capture(Thread):
    def __init__(self, control: Control, device: str, consumer: Callable[[Union[Tick, float]], None]):
        super().__init__()

        self._control: Control = control
        self._consumer: Callable[[Union[Tick, float]], None] = consumer
        self._inp: alsaaudio.PCM = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK,
            device=device,
            channels=1,
            rate=int(SAMPLE_RATE),
            format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=160,
        )

        self._highpass = signal.butter(HIGHPASS_ORDER, HIGHPASS_FREQ, 'hp', fs=SAMPLE_RATE, output='sos')

        self._running: bool = True
        self._last_timestamp_ms: int = -1
        self._within_MEM = 0

    def stop(self):
        self._running = False

    def _error(self, tick):
        print("E")
        if self._within_MEM == 1:
            self._within_MEM += 1
        self._control.error(tick.get_start_timestamp())

    def _miss(self, timestamp: float):
        print("M")
        if self._within_MEM == 0 or self._within_MEM == 2:
            self._within_MEM += 1
        self._control.miss(timestamp)
        self._consumer(timestamp)

    def _tick(self, tick: Tick):
        print("T")
        self._control.tick(tick.get_start_timestamp())
        self._consumer(tick)

    def _process(self, tick, timestamp_ms):
        if self._within_MEM == 3:
            # RESET
            print("Reset...")
            self._within_MEM = 0
            self._last_timestamp_ms = -1

        if self._last_timestamp_ms < 0:
            self._tick(tick)
            self._last_timestamp_ms = tick.get_start_timestamp()
            return

        dt = (tick.get_start_timestamp() - self._last_timestamp_ms)

        if dt < (1.-TIMESTAMP_FILTER) * self._control.get_mvmt_timescale_ms():
            self._error(tick)
        elif dt > (1.+TIMESTAMP_FILTER) * self._control.get_mvmt_timescale_ms():
            missed = round(dt/self._control.get_mvmt_timescale_ms()) - 1
            if missed > 0:
                for i in range(missed):
                    self._last_timestamp_ms += self._control.get_mvmt_timescale_ms()
                    self._miss(self._last_timestamp_ms)
                self._process(tick, timestamp_ms)

            else:
                self._error(tick)
        else:
            self._tick(tick)
            self._last_timestamp_ms = tick.get_start_timestamp()


    def run(self):

        timestamp_ms = 0
        no_signal_timestamp_ms = 0

        tick = Tick(self._control)
        while self._running:
            l, data = self._inp.read()
            if l == 0:
                time.sleep(.001)
                continue
            if len(data) == 0:
                continue

            arr = np.frombuffer(data[:(2*l)], dtype=np.int16)
            arr = signal.sosfilt(self._highpass, arr)

            timestamp_ms += l * MS_PER_FRAME
            if not tick.possibly_append(arr, timestamp_ms):
                self._process(tick, timestamp_ms)
                tick = Tick(self._control)
                no_signal_timestamp_ms = timestamp_ms
            elif timestamp_ms - no_signal_timestamp_ms > 1000.:
                print("No signal...")
                self._control.no_signal()
                no_signal_timestamp_ms = timestamp_ms
