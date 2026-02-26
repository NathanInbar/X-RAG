import json
from pathlib import Path
import time


class SimpleMiningProfiler:
	""" Note: Not thread-safe! """

	def __init__(self, result_path: Path, save_interval: int = 10):
		self.result_path = result_path
		self.save_interval = save_interval
		self.record_count = 0
		self.contents = {}
		if result_path.exists():
			with open(result_path) as fp:
				self.contents = json.load(fp)

	def _save(self):
		with open(self.result_path, "w") as fp:
			json.dump(self.contents, fp, indent=2)
	
	def start(self, category: str, data = {}):
		return SimpleProfilingItem(category, time.time(), self, data)

	def record(self, category: str, duration: float, data):
		# Record contents
		if not (category in self.contents):
			self.contents[category] = []
		self.contents[category].append((duration, data))

		# Maybe write to disk 
		self.record_count += 1
		if self.record_count >= self.save_interval:
			self.record_count = 0
			self._save()
	
	def category_mean_median(self):
		output = {}
		for cat, items in self.contents.items():
			if len(items) == 0:
				print(f"WARN: Profiling category '{cat}' has no entries, skipping")
				continue
			durations = [d for d, _ in items]
			mean = sum(durations) / len(durations)
			durations.sort()
			median = durations[len(durations)//2]
			output[cat] = (mean, median, durations)
		return output

	def finish(self):
		self._save()


class SimpleProfilingItem:
	def __init__(self, category: str, st: float, profiler, data):
		self.category = category
		self.st = st
		self.profiler = profiler
		self.data = data
	
	def end(self):
		self.profiler.record(self.category, time.time() - self.st, self.data)
