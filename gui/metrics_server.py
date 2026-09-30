import os
from threading import Event

from prometheus_client import CollectorRegistry, start_http_server
from prometheus_client.multiprocess import MultiProcessCollector


def main() -> None:
    registry = CollectorRegistry()
    MultiProcessCollector(registry)
    start_http_server(int(os.getenv("METRICS_PORT", "8502")), registry=registry)
    Event().wait()


if __name__ == "__main__":
    main()