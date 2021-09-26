from typing import Optional, Callable

from threading import Thread
import time
import alsaaudio
import numpy as np
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import librosa

SAMPLE_RATE = 44100.
MS_PER_FRAME = 1000. / SAMPLE_RATE


HIGHPASS_FREQ = 3000
HIGHPASS_ORDER = 10

TICK_PRERECORD = 0.15
TICK_MINLENGTH = 0.3
TICK_MAXLENGTH = 0.5

TICK_PEAK_FFT_SIZE = 64

class Control:
    def __init__(self):
        # TODO: Detect
        self.mvmt_bph: int = 18000

        # TODO: Control loop based on number of M / E
        self.tick_threshold: int = 1000

        # TODO: Control loop based on number of detected ticks per tick and quality of beat error
        # (missing the correct first tick often means beat error jumps a lot)
        self.peak_threshold: float = 0.01

    def get_mvmt_timescale_ms(self) -> float:
        return 3600000. / self.mvmt_bph

    def get_mvmt_timescale_frames(self) -> float:
        return self.get_mvmt_timescale_ms() / MS_PER_FRAME


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
        max_val = max(-next_input.min(), next_input.max())
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
        return np.arange(self._buffer.shape[0]) * MS_PER_FRAME, self._buffer

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

    def plot(self, line: mpl.lines.Line2D):
        if len(self._ticks) == 0:
            self._calculate_ticks()

        for l in list(line.axes.lines):
            if l != line:
                l.remove()

        start = self.get_start_timestamp() - self._timestamp
        final = self.get_final_timestamp() - self._timestamp

        t, y = self.get_wave()
        line.axes.set_xlim((start - 0.1*self._control.get_mvmt_timescale_ms(), start +
                            0.5*self._control.get_mvmt_timescale_ms()))
        lim = max(1000, max(-y.min(), y.max()))
        line.axes.set_ylim((-lim, lim))
        line.set_xdata(t)
        line.set_ydata(y)

        for t, _ in self._ticks:
            line.axes.axvline(x=t*MS_PER_FRAME, color='g', alpha=0.3)
        line.axes.axvline(x=start, color='r')
        line.axes.axvline(x=final, color='r')

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
        # 4. Score tick by absolute value of df_norms maximum

        peaks, _ = signal.find_peaks(df_norms, threshold=self._control.peak_threshold)
        for p in peaks:
            m = np.argmax(np.abs(self._buffer[((p+1)*TICK_PEAK_FFT_SIZE):((p+2)*TICK_PEAK_FFT_SIZE)]))
            self._ticks += [((p+1)*TICK_PEAK_FFT_SIZE + m, df_norms[p])]

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


class CaptureThread(Thread):
    def __init__(self, device: str, consumer: Callable[[Optional[Tick]], None]):
        super().__init__()

        self._control: Control = Control()
        self._consumer: Callable[[Optional[Tick]], None] = consumer
        self._inp: alsaaudio.PCM = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK,
            device=device,
            channels=1,
            rate=int(SAMPLE_RATE),
            format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=160,
        )

        self._highpass = signal.butter(HIGHPASS_ORDER, HIGHPASS_FREQ, 'hp', fs=SAMPLE_RATE, output='sos')
        self._timestamp_ms: int = 0

        self._running: bool = True

    def stop(self):
        self._running = False

    def _error(self):
        pass

    def _miss(self):
        self._consumer(None)

    def _tick(self, tick: Tick):
        self._consumer(tick)

    def run(self):
        tick = Tick(self._control)
        last_tick = tick
        while self._running:
            l, data = self._inp.read()
            if l == 0:
                time.sleep(.001)
                continue
            if len(data) == 0:
                continue
            arr = np.frombuffer(data[:(2*l)], dtype=np.int16)
            arr_filtered = signal.sosfilt(self._highpass, arr)

            self._timestamp_ms += l * MS_PER_FRAME
            if not tick.possibly_append(arr_filtered, self._timestamp_ms):
                dt = (tick._timestamp - last_tick._timestamp)
                if dt < 0.9 * self._control.get_mvmt_timescale_ms():
                    self._error()
                elif dt > 1.1 * self._control.get_mvmt_timescale_ms():
                    missed = round(dt/self._control.get_mvmt_timescale_ms()) - 1
                    for _ in range(missed):
                        self._miss()

                    last_timestamp = last_tick._timestamp + missed*self._control.get_mvmt_timescale_ms()
                    dt = tick._timestamp - last_timestamp
                    if dt < 0.9 * self._control.get_mvmt_timescale_ms():
                        self._error()
                    elif dt > 1.1 * self._control.get_mvmt_timescale_ms():
                        self._error()

                    self._tick(tick)
                    last_tick = tick
                else:
                    self._tick(tick)
                    last_tick = tick

                tick = Tick(self._control)

class Consumer:
    def __init__(self):
        self._at_tock: bool = False

        self._last_tick: Optional[Tick] = None
        self._last_tock: Optional[Tick] = None

    def __call__(self, tick: Tick) -> None:
        rate = -1 
        beat_error = -1
        dt_tick = -1
        dt_tock = -1

        if self._at_tock:
            if tick is not None and self._last_tick is not None:
                rate = tick.get_start_timestamp() - self._last_tick.get_start_timestamp()
            if tick is not None and self._last_tick is not None and self._last_tock is not None:
                beat_error = np.abs(tick.get_start_timestamp() - 2*self._last_tick.get_start_timestamp() +
                                    self._last_tock.get_start_timestamp())
            if tick is not None:
                dt_tock = tick.get_final_timestamp() - tick.get_start_timestamp()

            self._last_tock = tick
        else:
            if tick is not None and self._last_tock is not None:
                rate = tick.get_start_timestamp() - self._last_tock.get_start_timestamp()
            if tick is not None and self._last_tick is not None and self._last_tock is not None:
                beat_error = np.abs(tick.get_start_timestamp() - 2*self._last_tock.get_start_timestamp() +
                                    self._last_tick.get_start_timestamp())
            if tick is not None:
                dt_tick = tick.get_final_timestamp() - tick.get_start_timestamp()

            self._last_tick = tick
        self._at_tock = not self._at_tock

        print("Dt=%.2fms, BE=%.2fms, dt_1=%.2fms, dt_2=%.2fms" % (rate, beat_error, dt_tick, dt_tock))

if __name__ == '__main__':
    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # line1, = ax.plot([0, 1], [0, 0], 'b-')
    #
    # plot_counter = 0
    # def consumer(tick):
    #     global plot_counter
    #
    #     if tick is None:
    #         return
    #
    #     plot_counter += 1
    #     if plot_counter%10 == 0:
    #         tick.plot(line1)
    #         fig.canvas.draw()
    #         fig.canvas.flush_events()

    consumer = Consumer()

    CaptureThread("hw:2,0", consumer).start()
