"""Format-aware integrity validation for downloaded Drive files.

A download that reports success but yields a zero-filled or truncated file is
worse than a reported failure, because it silently enters the pipeline. This
module validates by container signature rather than by size alone.

Observed in this archive: several issue-262 page scans came back with the
correct byte length but 100% zero content, via two independent download paths.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

OLE_MAGIC = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'
PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
JPEG_SOI = b'\xff\xd8\xff'
JPEG_EOI = b'\xff\xd9'
RIFF_MAGIC = b'RIFF'
EPS_HEADER = b'%!PS-Adobe'
ZIP_MAGIC = b'PK\x03\x04'
PROBE_BYTES = 4096


def _tail(path: str, n: int = 64) -> bytes:
    size = os.path.getsize(path)
    with open(path, 'rb') as fh:
        fh.seek(max(0, size - n))
        return fh.read()


def _head(path: str, n: int = PROBE_BYTES) -> bytes:
    with open(path, 'rb') as fh:
        return fh.read(n)


def validate(path: str, ext: Optional[str] = None) -> Tuple[bool, str]:
    """Return (ok, reason). ok=False means the file must not be trusted.

    `ext` names the format to check against. Pass it whenever `path` is a
    temporary name (e.g. '<dest>.part'): the format is otherwise taken from
    the path's own extension, and '.part' matches no format, which silently
    skipped every format-specific check at download time until 2026-09-24.
    """
    if not os.path.exists(path):
        return False, 'missing'
    size = os.path.getsize(path)
    if size == 0:
        return False, 'zero_bytes'

    if ext is None:
        ext = os.path.splitext(path)[1]
    ext = ext.lower().lstrip('.')
    head = _head(path)

    # A file whose entire probed head is zero is never legitimate.
    if not any(head[:1024]):
        # confirm across the whole file for small files, else trust the head
        if size <= 1 << 20:
            data = open(path, 'rb').read()
            if not any(data):
                return False, 'all_zero_content'
        else:
            mid = _head_at(path, size // 2)
            if not any(head) and not any(mid) and not any(_tail(path, 1024)):
                return False, 'all_zero_content'

    if ext in ('jpg', 'jpeg'):
        if not head.startswith(JPEG_SOI):
            return False, 'jpeg_missing_soi'
        if JPEG_EOI not in _tail(path, 4096):
            return False, 'jpeg_missing_eoi_truncated'
        return True, 'ok'

    if ext == 'png':
        return (head.startswith(PNG_MAGIC), 'ok' if head.startswith(PNG_MAGIC)
                else 'png_bad_magic')

    if ext in ('tif', 'tiff'):
        ok = head[:4] in (b'II*\x00', b'MM\x00*')
        return ok, 'ok' if ok else 'tiff_bad_magic'

    if ext in ('inp', 'b01'):
        ok = head.startswith(OLE_MAGIC)
        return ok, 'ok' if ok else 'ole_bad_magic'

    if ext == 'cdr':
        ok = head.startswith(RIFF_MAGIC)
        return ok, 'ok' if ok else 'riff_bad_magic'

    if ext == 'eps':
        # EPS may carry a DOS binary preview header before the PostScript
        ok = EPS_HEADER in head[:1024] or head.startswith(b'\xc5\xd0\xd3\xc6')
        return ok, 'ok' if ok else 'eps_no_postscript_header'

    if ext in ('zip', 'docx'):
        ok = head.startswith(ZIP_MAGIC)
        return ok, 'ok' if ok else 'zip_bad_magic'

    return True, 'ok_unchecked_format'


def _head_at(path: str, offset: int, n: int = PROBE_BYTES) -> bytes:
    with open(path, 'rb') as fh:
        fh.seek(offset)
        return fh.read(n)


def describe(path: str) -> str:
    ok, reason = validate(path)
    return ('%s %s (%d bytes)' % ('OK  ' if ok else 'BAD ', reason,
                                  os.path.getsize(path)))
