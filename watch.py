import subprocess
import time
import sys
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

file_hashes = {}

def hash_file(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

class TestRunnerHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            current_hash = hash_file(event.src_path)
            previous_hash = file_hashes.get(event.src_path)

            if current_hash and current_hash != previous_hash:
                file_hashes[event.src_path] = current_hash
                print(f'\n🌀 Real change detected in {event.src_path}, running tests...\n')
                subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_question.py', '-v'])

if __name__ == "__main__":
    path = "."
    event_handler = TestRunnerHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    print("👀 Watching for changes in .py files... (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
