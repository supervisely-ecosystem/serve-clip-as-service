from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Dict, List, TextIO
import atexit

import numpy as np
import supervisely as sly
from clip_client import Client

from src.utils import timeit, with_retries, read_yaml


CAS_CONFIG_PATH = Path(__file__).parent / "cas.yml"


def get_cas_config():
    return read_yaml(CAS_CONFIG_PATH)


def get_port(cas_config: Dict):
    return cas_config["with"]["port"]


CAS_CONFIG = get_cas_config()
CAS_PORT = get_port(CAS_CONFIG)
CAS_HOST = f"0.0.0.0:{CAS_PORT}"
CAS_START_TIME = 10 * 60
cas_started = threading.Event()


def _process_lines(io: TextIO, line_callbacks=None):
    for line in iter(io.readline, ""):
        for cb in line_callbacks:
            cb(line)


def _set_started(line: str):
    if "Flow is ready to serve!" in line:
        cas_started.set()


def _print_line(line: str):
    sys.stdout.write(line)
    sys.stdout.flush()


def start_server() -> int:
    pcs = subprocess.Popen(
        ["python3", "-m", "clip_server"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
    )
    atexit.register(pcs.terminate)
    threading.Thread(target=_process_lines, args=[pcs.stdout, [_set_started, _print_line]], daemon=True).start()


def wait_server_for_start():
    t = time.monotonic()
    while time.monotonic() < t + CAS_START_TIME:
        if cas_started.is_set():
            return
        time.sleep(1)
    raise RuntimeError(f"Unable to start CAS server in time ({CAS_START_TIME // 60} minutes)")


def run_server():
    start_server()
    wait_server_for_start()


def get_client(cas_host=CAS_HOST) -> Client:
    client = Client(f"grpc://{cas_host}")
    try:
        sly.logger.info(f"Connecting to CAS at {cas_host}...")
        client.profile()
        sly.logger.info(f"Connected to CAS at {cas_host}")
    except Exception as e:
        sly.logger.error(f"Failed to connect to CAS at {cas_host}: {e}")
    return client


@with_retries(retries=5, sleep_time=2)
@timeit
async def get_vectors(client: Client, queries: List[str]) -> List[np.ndarray]:
    """Use CAS to get vectors from the list of queries.
    List of queries is a list of URLs for images or text prompts.

    :param queries: List of queries (URLs for images or text prompts).
    :type queries: List[str]
    :return: List of vectors.
    :rtype: List[np.ndarray]
    """
    vectors = await client.aencode(queries)
    return [vector.tolist() for vector in vectors]
