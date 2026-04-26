from __future__ import annotations

import collections
import csv
import math
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
import psutil
import pyaudio

try:
    from pynput import keyboard
except ImportError:
    keyboard = None

try:
    from rich import box
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    print("pip install rich")
    sys.exit(1)

console = Console()


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE_HZ = 10
SAMPLE_PERIOD_S = 1.0 / SAMPLE_RATE_HZ
RING_BUFFER_SIZE = 200
DISPLAY_HISTORY = 20
CAMERA_INDEXES = (0, 1, 2)
MIC_RATE = 16_000
MIC_FRAMES = MIC_RATE // SAMPLE_RATE_HZ
ANOMALY_SIGMA = 3.2
ANOMALY_MIN_SAMPLES = 25
ANOMALY_COOLDOWN_S = 1.5


# ============================================================
# RING BUFFER
# ============================================================

class SensorChannel:
    """One channel of the DAQ: value, timestamp, history."""

    def __init__(self, name: str, unit: str, color: str = "white"):
        self.name = name
        self.unit = unit
        self.color = color
        self.values = collections.deque(maxlen=RING_BUFFER_SIZE)
        self.timestamps = collections.deque(maxlen=RING_BUFFER_SIZE)
        self.lock = threading.Lock()

    def push(self, value: float, timestamp: Optional[float] = None) -> None:
        stamp = time.time() if timestamp is None else timestamp
        with self.lock:
            self.values.append(float(value))
            self.timestamps.append(stamp)

    def latest(self) -> float:
        with self.lock:
            return self.values[-1] if self.values else 0.0

    def history(self, n: int = DISPLAY_HISTORY) -> List[float]:
        with self.lock:
            return list(self.values)[-n:]

    def sample_count(self) -> int:
        with self.lock:
            return len(self.values)

    def mean(self) -> float:
        with self.lock:
            vals = list(self.values)
        return float(np.mean(vals)) if vals else 0.0

    def std(self) -> float:
        with self.lock:
            vals = list(self.values)
        return float(np.std(vals)) if len(vals) > 1 else 0.0


# ============================================================
# SENSOR CHANNELS
# ============================================================

channels: Dict[str, SensorChannel] = {
    "mic_rms": SensorChannel("Mic RMS", "amplitude", "cyan"),
    "cam_motion": SensorChannel("Cam Motion", "px/frame", "green"),
    "cpu_pct": SensorChannel("CPU %", "%", "yellow"),
    "mem_pct": SensorChannel("Memory %", "%", "magenta"),
    "keystroke_rate": SensorChannel("Keys/sec", "Hz", "blue"),
}


# ============================================================
# SENSOR READERS
# ============================================================

running = True
keystroke_counter = 0
keystroke_lock = threading.Lock()
anomaly_log = collections.deque(maxlen=5)
last_alert_times: Dict[str, float] = {}


def sleep_until(next_tick: float) -> float:
    now = time.time()
    delay = max(0.0, next_tick - now)
    if delay:
        time.sleep(delay)
    return next_tick + SAMPLE_PERIOD_S


def find_input_device(pa: pyaudio.PyAudio) -> Optional[int]:
    for index in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(index)
        if info.get("maxInputChannels", 0) > 0:
            return index
    return None


def mic_reader() -> None:
    next_tick = time.time()
    try:
        pa = pyaudio.PyAudio()
        device_index = find_input_device(pa)
        if device_index is None:
            raise RuntimeError("No microphone input device found")
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=MIC_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=MIC_FRAMES,
        )
    except Exception:
        while running:
            channels["mic_rms"].push(0.0)
            next_tick = sleep_until(next_tick)
        return

    try:
        while running:
            try:
                data = stream.read(MIC_FRAMES, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.float32)
                rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
                channels["mic_rms"].push(rms)
            except Exception:
                channels["mic_rms"].push(0.0)
            next_tick = sleep_until(next_tick)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def open_camera() -> Optional[cv2.VideoCapture]:
    for index in CAMERA_INDEXES:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def cam_reader() -> None:
    next_tick = time.time()
    cap = open_camera()
    if cap is None:
        while running:
            channels["cam_motion"].push(0.0)
            next_tick = sleep_until(next_tick)
        return

    prev_gray = None
    try:
        while running:
            ok, frame = cap.read()
            if not ok or frame is None:
                channels["cam_motion"].push(0.0)
                next_tick = sleep_until(next_tick)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 120))

            if prev_gray is None:
                magnitude = 0.0
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    None,
                    0.5,
                    3,
                    15,
                    3,
                    5,
                    1.2,
                    0,
                )
                magnitude = float(np.mean(np.linalg.norm(flow, axis=2)))

            channels["cam_motion"].push(magnitude)
            prev_gray = gray
            next_tick = sleep_until(next_tick)
    finally:
        cap.release()


def keyboard_listener_worker() -> None:
    global keystroke_counter

    if keyboard is None:
        return

    def on_press(_key) -> None:
        global keystroke_counter
        with keystroke_lock:
            keystroke_counter += 1

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    while running:
        time.sleep(0.1)
    listener.stop()


def keystroke_reader() -> None:
    global keystroke_counter
    next_tick = time.time()
    while running:
        with keystroke_lock:
            count = keystroke_counter
            keystroke_counter = 0
        channels["keystroke_rate"].push(count * SAMPLE_RATE_HZ)
        next_tick = sleep_until(next_tick)


def system_reader() -> None:
    next_tick = time.time()
    psutil.cpu_percent(interval=None)
    while running:
        channels["cpu_pct"].push(psutil.cpu_percent(interval=None))
        channels["mem_pct"].push(psutil.virtual_memory().percent)
        next_tick = sleep_until(next_tick)


# ============================================================
# ANOMALY DETECTION
# ============================================================

def detect_anomaly(name: str, ch: SensorChannel) -> Optional[str]:
    count = ch.sample_count()
    if count < ANOMALY_MIN_SAMPLES:
        return None

    val = ch.latest()
    mean = ch.mean()
    std = ch.std()
    if std <= 1e-6:
        return None

    z_score = abs(val - mean) / std
    if z_score < ANOMALY_SIGMA:
        return None

    now = time.time()
    if now - last_alert_times.get(name, 0.0) < ANOMALY_COOLDOWN_S:
        return None

    last_alert_times[name] = now
    direction = "up" if val > mean else "down"
    return f"{ch.name} {direction} {val:.3f} ({z_score:.1f} sigma)"


def check_anomalies() -> List[str]:
    alerts = []
    for name, ch in channels.items():
        alert = detect_anomaly(name, ch)
        if alert:
            alerts.append(alert)
    return alerts


# ============================================================
# SPARKLINE
# ============================================================

def sparkline(values: List[float], width: int = 20) -> str:
    """ASCII sparkline from a list of values."""
    chars = " ▁▂▃▄▅▆▇█"
    if not values:
        return "─" * width

    window = values[-width:]
    lo, hi = min(window), max(window)
    if math.isclose(lo, hi):
        return "─" * width

    result = []
    span = hi - lo
    for value in window:
        idx = int((value - lo) / span * (len(chars) - 1))
        result.append(chars[idx])
    return "".join(result).ljust(width)


# ============================================================
# RICH DASHBOARD
# ============================================================

def build_dashboard() -> Panel:
    table = Table(
        title="SensorLogger - DAQ Dashboard",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Channel", style="bold", width=14)
    table.add_column("Latest", justify="right", width=14)
    table.add_column("Mean", justify="right", width=10)
    table.add_column("Std", justify="right", width=10)
    table.add_column("Trend", width=22)
    table.add_column("Status", width=10)

    for name, ch in channels.items():
        val = ch.latest()
        mean = ch.mean()
        std = ch.std()
        hist = ch.history()
        sample_count = ch.sample_count()
        is_anomaly = (
            sample_count >= ANOMALY_MIN_SAMPLES
            and std > 1e-6
            and abs(val - mean) / std >= ANOMALY_SIGMA
        )
        status = "[red]SPIKE[/red]" if is_anomaly else "[green]OK[/green]"
        table.add_row(
            f"[{ch.color}]{ch.name}[/{ch.color}]",
            f"[{ch.color}]{val:.3f}[/{ch.color}] {ch.unit}",
            f"{mean:.3f}",
            f"{std:.3f}",
            f"[{ch.color}]{sparkline(hist)}[/{ch.color}]",
            status,
        )

    alerts = check_anomalies()
    if alerts:
        anomaly_log.extend(alerts)

    stats_text = (
        f"[dim]Samples: {min(ch.sample_count() for ch in channels.values())} | "
        f"Buffer: {RING_BUFFER_SIZE} | "
        f"Rate: {SAMPLE_RATE_HZ} Hz | "
        f"Press 's' then Enter to export CSV, 'q' then Enter to quit[/dim]"
    )
    if anomaly_log:
        alert_text = "[red]Recent anomalies: " + " | ".join(list(anomaly_log)[-3:]) + "[/red]"
    else:
        alert_text = "[green]No anomalies detected[/green]"

    return Panel(table, subtitle=f"{stats_text}\n{alert_text}", border_style="cyan")


# ============================================================
# CSV EXPORT
# ============================================================

def export_csv() -> str:
    """Export all channel data to a timestamped CSV."""
    filename = f"sensorlog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    min_len = min(ch.sample_count() for ch in channels.values())

    with open(filename, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", *channels.keys()])

        timestamp_rows = [list(ch.timestamps)[:min_len] for ch in channels.values()]
        value_rows = [list(ch.values)[:min_len] for ch in channels.values()]

        for idx in range(min_len):
            timestamp = timestamp_rows[0][idx]
            row = [datetime.fromtimestamp(timestamp).isoformat(timespec="milliseconds")]
            row.extend(values[idx] for values in value_rows)
            writer.writerow(row)

    console.print(f"\n[green]Exported {filename} ({min_len} rows)[/green]")
    return filename


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    global running

    console.print("\n[bold cyan]SensorLogger - Day 20[/bold cyan]")
    console.print("[dim]Starting sensor threads...[/dim]\n")

    threads = [
        threading.Thread(target=mic_reader, daemon=True),
        threading.Thread(target=cam_reader, daemon=True),
        threading.Thread(target=system_reader, daemon=True),
        threading.Thread(target=keystroke_reader, daemon=True),
    ]

    if keyboard is not None:
        threads.append(threading.Thread(target=keyboard_listener_worker, daemon=True))
    else:
        console.print("[yellow]pynput not installed. Keystroke channel will stay at 0.[/yellow]")

    for thread in threads:
        thread.start()

    time.sleep(1.5)
    console.print("[green]Sensors active[/green]")
    console.print("[dim]Press 's' + Enter to export CSV. Press 'q' + Enter to quit.[/dim]\n")

    export_flag = threading.Event()

    def input_listener() -> None:
        global running
        while running:
            try:
                cmd = input().strip().lower()
            except EOFError:
                break
            except Exception:
                continue

            if cmd == "s":
                export_flag.set()
            elif cmd == "q":
                running = False
                break

    input_thread = threading.Thread(target=input_listener, daemon=True)
    input_thread.start()

    try:
        with Live(build_dashboard(), refresh_per_second=4, console=console) as live:
            while running:
                live.update(build_dashboard())
                if export_flag.is_set():
                    export_csv()
                    export_flag.clear()
                time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        running = False

    console.print("\n[bold]SensorLogger ended.[/bold]")
    console.print("See you tomorrow for Day 21!")


if __name__ == "__main__":
    main()
