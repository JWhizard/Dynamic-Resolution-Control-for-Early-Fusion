"""Sample named INA3221 rails; never treat overlapping shunt sums as power."""
from pathlib import Path
import threading
import time
import numpy as np


class RailSampler:
    def __init__(self, root='/sys/class/hwmon', interval=.1):
        self.rails = {}
        for hw in Path(root).glob('hwmon*'):
            if (hw / 'name').read_text().strip() != 'ina3221':
                continue
            for label in hw.glob('in*_label'):
                channel = label.stem[2:].split('_')[0]
                current = hw / f'curr{channel}_input'
                if current.exists():
                    name = label.read_text().strip()
                    self.rails[name] = (hw / f'in{channel}_input', current)
        # AGX Orin exposes these disjoint named rails rather than a VDD_IN rail.
        # Keep rail-level records; their sum is not wall-plug energy.
        self.interval = interval
        self.samples = []
        self.errors = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = None

    def sample(self):
        values = {}
        for name, (voltage, current) in self.rails.items():
            try:
                values[name] = float(voltage.read_text()) * float(current.read_text()) / 1e6
            except (OSError, ValueError, TypeError) as exc:
                self.errors.append(str(exc))
        row = {'time': time.monotonic(), 'watts': values}
        with self.lock:
            self.samples.append(row)
        return row

    def start(self):
        self.sample()
        def work():
            while not self.stop_event.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.sample()

    def energy(self, start, end):
        with self.lock:
            rows = list(self.samples)
        result = {}
        for name in self.rails:
            points = [(r['time'], r['watts'][name]) for r in rows if name in r['watts']]
            if len(points) < 2 or points[0][0] > start or points[-1][0] < end:
                result[name] = None
                continue
            t, p = map(np.asarray, zip(*points))
            inside = (t > start) & (t < end)
            times = np.r_[start, t[inside], end]
            powers = np.r_[np.interp(start, t, p), p[inside], np.interp(end, t, p)]
            result[name] = float(np.trapz(powers, times))
        return result
