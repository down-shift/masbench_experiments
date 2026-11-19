"""Shared file and evaluation conventions."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parent
HISTORY = "MASBENCH_EXPERIMENT_HISTORY_PLACEHOLDER_8A501E"
AGENT_FORMAT = (
    "\nWhen you actively propose a time, put each proposed time on its own line "
    "as PROPOSE Day X HH:MM-HH:MM. Do not use PROPOSE for rejected slots or examples. "
    "You can discuss freely outside those lines.\n"
)
LEADER_FORMAT = (
    "\nYou are the final decision-maker. Use the discussion to choose one meeting "
    "that works for all five participants. Output exactly one line: "
    "FINAL Day X HH:MM-HH:MM, or NO_MEETING if you cannot choose.\n"
)
SLOT = r"Day ([1-5]) (\d{2}):(\d{2})[-–](\d{2}):(\d{2})"


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    """Commit a complete JSON file, including after an interrupted run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as out:
        temp = Path(out.name)
        try:
            json.dump(value, out, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def slot(line, label):
    match = re.fullmatch(label + " " + SLOT, line.strip())
    if not match:
        return None
    day, h, m, eh, em = map(int, match.groups())
    begin, end = h * 60 + m, eh * 60 + em
    if h >= 24 or m >= 60 or eh > 24 or em >= 60 or begin % 10 or end > 1440 or end - begin != 20:
        return None
    return (day - 1) * 144 + begin // 10


def format_slot(index):
    day, unit = divmod(index, 144)
    start, end = unit * 10, unit * 10 + 20
    require(end <= 1440, "The textual format cannot represent a cross-day meeting.")
    return f"Day {day + 1} {start // 60:02}:{start % 60:02}-{end // 60:02}:{end % 60:02}"
