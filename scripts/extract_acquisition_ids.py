"""Extract ICEYE acquisition IDs from Thunderbird email subjects.

Reads a Thunderbird mbox file (e.g. the ``Download_Image`` subfolder of
``INBOX`` on a Gmail IMAP account) and prints/saves every numeric ID found in
subjects of the form::

    Acquisition 9943920 is ready for download

Usage examples::

    # Use the default mbox path (Snap Thunderbird, INBOX/Download_Image)
    sudo python scripts/extract_acquisition_ids.py

    # Specify a different mbox path
    sudo python scripts/extract_acquisition_ids.py --mbox /path/to/folder

    # Save the IDs to a text file (one per line, deduplicated, sorted)
    sudo python scripts/extract_acquisition_ids.py -o acquisition_ids.txt

Note:
    The Thunderbird profile lives under ``~/snap/thunderbird/common/...`` and
    is root-owned, so the script normally needs to be run with ``sudo``.
    Alternatively, copy the mbox file to a readable location first and pass
    it via ``--mbox``.
"""

from __future__ import annotations

import argparse
import mailbox
import re
import sys
from datetime import datetime, timezone
from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime
from pathlib import Path

DEFAULT_MBOX = Path(
    "/home/odogan/snap/thunderbird/common/.thunderbird/"
    "5kj9dux2.default/ImapMail/imap.gmail.com/INBOX.sbd/Download_Image"
)

SUBJECT_RE = re.compile(r"Acquisition\s+(\d+)\s+is\s+ready\s+for\s+download", re.IGNORECASE)

EXPUNGED_FLAG = 0x0008
IMAP_DELETED_FLAG = 0x00200000


def decode_subject(raw: str | None) -> str:
    """Decode a MIME-encoded ``Subject:`` header into a plain string."""
    if not raw:
        return ""
    try:
        return str(make_header(decode_header(raw)))
    except Exception:
        return raw


def _hex_header(message, name: str) -> int:
    """Parse an ``X-Mozilla-Status*`` header value (hex, no prefix) into int."""
    raw = message.get(name)
    if not raw:
        return 0
    try:
        return int(raw.strip().split()[0], 16)
    except (ValueError, IndexError):
        return 0


def parse_since(value: str) -> datetime:
    """Parse a ``--since`` argument.

    Accepts ``DD/MM/YYYY``, ``YYYY-MM-DD``, ``DD-MM-YYYY``. The returned
    datetime is timezone-aware (UTC, midnight at the start of the day).
    """
    formats = ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y")
    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Could not parse date {value!r}. Use DD/MM/YYYY (e.g. 16/06/2026)."
    )


def message_date(message) -> datetime | None:
    """Return the ``Date:`` header as a timezone-aware datetime, or None."""
    raw = message.get("Date")
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def is_deleted(message) -> bool:
    """Return True if Thunderbird has flagged the message as deleted.

    Thunderbird keeps deleted messages in the mbox file until the folder is
    compacted. They are tagged via the ``X-Mozilla-Status`` (Expunged bit
    ``0x0008``) and ``X-Mozilla-Status2`` (IMAPDeleted bit ``0x00200000``)
    headers.
    """
    status = _hex_header(message, "X-Mozilla-Status")
    status2 = _hex_header(message, "X-Mozilla-Status2")
    return bool(status & EXPUNGED_FLAG) or bool(status2 & IMAP_DELETED_FLAG)


def iter_messages(mbox_path: Path):
    """Yield ``(key, message)`` pairs from the mbox file."""
    box = mailbox.mbox(str(mbox_path))
    try:
        for key, message in box.iteritems():
            yield key, message
    finally:
        box.close()


def extract_ids(
    mbox_path: Path,
    since: datetime | None = None,
    skip_deleted: bool = True,
    verbose: bool = False,
):
    """Walk the mbox file and return ``(ids, unmatched_subjects, stats)``.

    Messages older than ``since`` (inclusive of the start of that day) are
    skipped. Messages flagged as deleted by Thunderbird are also skipped
    when ``skip_deleted`` is True. ``ids`` preserves insertion order but is
    deduplicated.
    """
    seen: set[str] = set()
    ids: list[str] = []
    unmatched: list[str] = []
    total = 0
    n_deleted = 0
    n_too_old = 0

    for _key, message in iter_messages(mbox_path):
        total += 1
        if skip_deleted and is_deleted(message):
            n_deleted += 1
            if verbose:
                subject = decode_subject(message.get("Subject"))
                print(f"[del ] {subject}")
            continue

        if since is not None:
            dt = message_date(message)
            if dt is None or dt < since:
                n_too_old += 1
                if verbose:
                    subject = decode_subject(message.get("Subject"))
                    print(f"[old ] {dt}  <- {subject}")
                continue

        subject = decode_subject(message.get("Subject"))
        match = SUBJECT_RE.search(subject)
        if match:
            acq_id = match.group(1)
            if acq_id not in seen:
                seen.add(acq_id)
                ids.append(acq_id)
            if verbose:
                print(f"[ok  ] {acq_id}  <- {subject}")
        else:
            unmatched.append(subject)
            if verbose:
                print(f"[skip] {subject}")

    print(
        f"\nScanned {total} messages:"
        f" skipped {n_deleted} deleted,"
        f" {n_too_old} older than cutoff,"
        f" matched {len(ids)} unique IDs"
        f" ({len(unmatched)} eligible subjects did not match).",
        file=sys.stderr,
    )
    return ids, unmatched, {"deleted": n_deleted, "too_old": n_too_old}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mbox",
        type=Path,
        default=DEFAULT_MBOX,
        help=f"Path to the Thunderbird mbox file (default: {DEFAULT_MBOX}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output file. If given, IDs are sorted and written one per line.",
    )
    parser.add_argument(
        "--since",
        type=parse_since,
        default=None,
        help=(
            "Only include emails on/after this date."
            " Accepts DD/MM/YYYY, YYYY-MM-DD, DD-MM-YYYY, DD.MM.YYYY."
        ),
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Do NOT skip messages flagged as deleted by Thunderbird.",
    )
    parser.add_argument(
        "--keep-order",
        action="store_true",
        help="Write IDs in mailbox order instead of sorted numerically.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print every subject as it is processed.",
    )
    parser.add_argument(
        "--show-unmatched",
        action="store_true",
        help="Print subjects that did not match the expected pattern.",
    )
    args = parser.parse_args()

    if not args.mbox.exists():
        print(f"error: mbox file not found: {args.mbox}", file=sys.stderr)
        return 2

    ids, unmatched, _stats = extract_ids(
        args.mbox,
        since=args.since,
        skip_deleted=not args.include_deleted,
        verbose=args.verbose,
    )

    if not args.keep_order:
        ids = sorted(set(ids), key=lambda s: int(s))

    joined = ",".join(ids)

    if args.output is None:
        print(joined)
    else:
        args.output.write_text(joined)
        print(f"Wrote {len(ids)} IDs to {args.output}", file=sys.stderr)

    if args.show_unmatched and unmatched:
        print("\n-- Unmatched subjects --", file=sys.stderr)
        for s in unmatched:
            print(s, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
