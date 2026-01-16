"""GPU 모니터링 유틸리티"""
import subprocess
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, List
import re


@dataclass
class GPUStats:
    """GPU 통계"""
    timestamp: float
    gpu_util: float  # %
    memory_used: float  # MB
    memory_total: float  # MB
    temperature: float  # C
    power_draw: float  # W
    power_cap: float  # W


class GPUMonitor:
    """실시간 GPU 모니터링"""

    def __init__(self, interval: float = 0.5, history_size: int = 100):
        self.interval = interval
        self.history: deque = deque(maxlen=history_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _parse_nvidia_smi(self) -> Optional[GPUStats]:
        """nvidia-smi 출력 파싱"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return None

            parts = result.stdout.strip().split(", ")
            if len(parts) >= 6:
                return GPUStats(
                    timestamp=time.time(),
                    gpu_util=float(parts[0]),
                    memory_used=float(parts[1]),
                    memory_total=float(parts[2]),
                    temperature=float(parts[3]),
                    power_draw=float(parts[4]) if parts[4] != "[N/A]" else 0,
                    power_cap=float(parts[5]) if parts[5] != "[N/A]" else 240
                )
        except Exception as e:
            print(f"nvidia-smi error: {e}")
        return None

    def _monitor_loop(self):
        """모니터링 루프"""
        while self._running:
            stats = self._parse_nvidia_smi()
            if stats:
                self.history.append(stats)
            time.sleep(self.interval)

    def start(self):
        """모니터링 시작"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("GPU monitoring started")

    def stop(self):
        """모니터링 중지"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("GPU monitoring stopped")

    def get_current(self) -> Optional[GPUStats]:
        """현재 통계"""
        if self.history:
            return self.history[-1]
        return self._parse_nvidia_smi()

    def get_summary(self) -> dict:
        """통계 요약"""
        if not self.history:
            return {}

        stats_list = list(self.history)
        return {
            "samples": len(stats_list),
            "gpu_util": {
                "avg": sum(s.gpu_util for s in stats_list) / len(stats_list),
                "max": max(s.gpu_util for s in stats_list),
                "min": min(s.gpu_util for s in stats_list),
            },
            "memory_used": {
                "avg": sum(s.memory_used for s in stats_list) / len(stats_list),
                "max": max(s.memory_used for s in stats_list),
            },
            "temperature": {
                "avg": sum(s.temperature for s in stats_list) / len(stats_list),
                "max": max(s.temperature for s in stats_list),
            },
            "power_draw": {
                "avg": sum(s.power_draw for s in stats_list) / len(stats_list),
                "max": max(s.power_draw for s in stats_list),
            },
        }

    def print_status(self):
        """현재 상태 출력"""
        stats = self.get_current()
        if stats:
            print(f"GPU: {stats.gpu_util:.0f}% | "
                  f"Mem: {stats.memory_used:.0f}/{stats.memory_total:.0f}MB ({100*stats.memory_used/stats.memory_total:.1f}%) | "
                  f"Temp: {stats.temperature:.0f}C | "
                  f"Power: {stats.power_draw:.0f}/{stats.power_cap:.0f}W")

    def print_summary(self):
        """요약 출력"""
        summary = self.get_summary()
        if summary:
            print("\n" + "=" * 50)
            print("GPU Monitoring Summary")
            print("=" * 50)
            print(f"Samples: {summary['samples']}")
            print(f"GPU Util: avg={summary['gpu_util']['avg']:.1f}%, "
                  f"max={summary['gpu_util']['max']:.1f}%, "
                  f"min={summary['gpu_util']['min']:.1f}%")
            print(f"Memory: avg={summary['memory_used']['avg']:.0f}MB, "
                  f"max={summary['memory_used']['max']:.0f}MB")
            print(f"Temp: avg={summary['temperature']['avg']:.1f}C, "
                  f"max={summary['temperature']['max']:.1f}C")
            print(f"Power: avg={summary['power_draw']['avg']:.1f}W, "
                  f"max={summary['power_draw']['max']:.1f}W")
            print("=" * 50)


# 전역 모니터 인스턴스
_monitor: Optional[GPUMonitor] = None


def start_monitoring(interval: float = 0.5):
    """전역 모니터링 시작"""
    global _monitor
    _monitor = GPUMonitor(interval=interval)
    _monitor.start()
    return _monitor


def stop_monitoring():
    """전역 모니터링 중지 및 요약 출력"""
    global _monitor
    if _monitor:
        _monitor.stop()
        _monitor.print_summary()
        return _monitor.get_summary()
    return None


def print_gpu_status():
    """현재 GPU 상태 출력"""
    if _monitor:
        _monitor.print_status()
    else:
        m = GPUMonitor()
        m.print_status()


if __name__ == "__main__":
    # 테스트
    print("Testing GPU monitor...")
    monitor = start_monitoring(interval=0.5)

    print("\nMonitoring for 5 seconds...")
    for i in range(10):
        time.sleep(0.5)
        monitor.print_status()

    stop_monitoring()
