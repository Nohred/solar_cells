import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def record_event(event: str, **fields: Any) -> None:
	"""Append a small operational event without persisting image contents."""
	path = Path(os.getenv("AUDIT_PATH", "/data/audit.jsonl"))
	path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"event": event,
		**fields,
	}
	try:
		with path.open("a", encoding="utf-8") as audit_file:
			audit_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
	except OSError:
		return
