#!/usr/bin/env python3
# Confidential, Copyright 2024, Sony AI, All rights reserved
"""
Ace Logs Watchdog utility tool
This tool is aimed to sync the local logs into an archieve folder (e.g. /mnt/sai-ace/)
by automatic detection to any new logs which were created and closed.
"""

import time
import logging
import queue
import os
import socket
import threading
import subprocess
import re
from enum import Enum
import psutil
import click
import colored_glog as glog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from watchdog.events import FileSystemEvent


class LogFilesHandler(FileSystemEventHandler):
    """File handler class, collects new logs and performs rsync to a destination folder"""

    class SyncTool(Enum):
        """type of sync tools"""

        RSYNC = 1
        RCLONE = 2

    def __init__(self, source_path, destination_path, sync_tool, rclone_remote, extension_regex) -> None:
        super().__init__()
        self.sync_queue: queue.Queue = queue.Queue()
        self.is_done = threading.Event()
        self.sync_thread = None
        self.source_path = source_path
        self.destination_path = destination_path
        self.extension_regex = extension_regex
        glog.info(f"Extensions to Monitor: {extension_regex}")
        glog.info(f"Source Path: {source_path}")
        glog.info(f"Destination Path: {destination_path}")

        if sync_tool.lower() == "rsync":
            self.sync_tool = LogFilesHandler.SyncTool.RSYNC
        elif sync_tool.lower() == "rclone":
            self.sync_tool = LogFilesHandler.SyncTool.RCLONE
            self.rclone_remote = rclone_remote
        else:
            raise Exception(f"Unkown syncronization tool type: {sync_tool}")

        self.sync_map = {}
        self.sync_map[LogFilesHandler.SyncTool.RSYNC] = self.rsync
        self.sync_map[LogFilesHandler.SyncTool.RCLONE] = self.rclone

        self.start_worker()

    def start_worker(self):
        """Start worker thread"""
        self.is_done.clear()
        self.sync_thread = threading.Thread(target=self._worker_thread)
        self.sync_thread.start()

    def rsync(self, src_file, destination):  # pylint: disable=no-self-use
        """rsync function"""
        invocation = f'rsync -z "{src_file}" "{destination}"'
        return invocation

    def rclone(self, src_file, destination):
        """rclone function"""
        invocation = f'rclone sync --bwlimit 1M "{src_file}" {self.rclone_remote}:"{destination}"'
        return invocation

    def execute_rsync(self, src_file: str):
        """Perform sync for a file"""
        if src_file.startswith(self.source_path):
            destination = os.path.join(self.destination_path, src_file[len(self.source_path) :])
        else:
            destination = os.path.join(self.destination_path, src_file)

        destination = os.path.dirname(destination)
        invocation = self.sync_map[self.sync_tool](src_file, destination)
        os.makedirs(destination, exist_ok=True)
        glog.info(f"Executing: {invocation}")
        subprocess.run(invocation, shell=True, check=False)
        glog.info(f"Done uploading: {src_file}")

    def close_worker(self):
        """Close worker thread"""
        self.is_done.set()
        self.sync_thread.join()
        self.sync_thread = None

    def _worker_thread(self):
        """Main thread"""
        while not self.is_done.is_set():
            while not self.sync_queue.empty():
                file = self.sync_queue.get()
                try:
                    self.execute_rsync(file)
                except Exception as err:  # pylint: disable=broad-except
                    glog.error(f"Failed to synchronize `{file}` with error: {err}")
            time.sleep(1)

    def on_closed(self, event: FileSystemEvent) -> None:
        """Callback from watchdog that a file was closed"""
        if re.match(self.extension_regex, event.src_path) is not None:
            self.sync_queue.put(event.src_path)


@click.command()
@click.option(
    "--src",
    default="",
    help="Source path to synchronize the logs from.",
    type=str,
)
@click.option(
    "--dst",
    default="/mnt/sai-ace/ace_logs/",
    help="Destination path to synchronize the logs into.",
    type=str,
)
@click.option(
    "--sync-tool",
    default="rsync",
    help="Synchronization tool type",
    type=str,
)
@click.option(
    "--rclone-remote",
    default="gdrive_teamdrive_ace",
    help="When using rclone, specify remote name",
    type=str,
)
@click.option(
    "--host-name",
    default="",
    help="Host name.",
    type=str,
)
@click.option(
    "--ext",
    multiple=True,
    default=["ace", "db3", "yaml", "mp4"],
    help="File extensions to monitor. Pass each extension as a separate argument,"
    + " e.g., --ext ace --ext db3 --ext yaml --ext mp4",
)
def main(src, dst, sync_tool, rclone_remote, host_name, ext):
    """Main entry point"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if src == "":
        if "ACE_DATALOGGER_PATH" in os.environ:
            src = os.environ["ACE_DATALOGGER_PATH"]
        else:
            raise ValueError("src should be specified to a path, or $ACE_DATALOGGER_PATH should be set globally")

    if host_name == "":
        host_name = socket.gethostname()
    destination = os.path.join(dst, host_name)
    ext_regex = ".*\\." + "|.*\\.".join(list(map(re.escape, ext)))

    # pin to core 1
    psutil.Process(os.getpid()).cpu_affinity([1])
    event_handler = LogFilesHandler(src, destination, sync_tool, rclone_remote, ext_regex)
    observer = Observer()
    observer.schedule(event_handler, src, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        observer.stop()
        event_handler.close_worker()
    observer.join()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
