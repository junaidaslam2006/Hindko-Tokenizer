"""Plain-text extraction from .docx (Office Open XML), stdlib only.

The book collection holds two .docx entries. One is a real document
(22nd Farma .../Index.docx). The other, '~$glish content.docx', is a Word
*owner/lock file*: Word writes a tiny '~$'-prefixed file next to a document
while it is open, holding only the editing user's name. It is not a ZIP
container and carries no document text, so it is classified, not extracted.
"""
from __future__ import annotations

import os
import re
import zipfile
from xml.etree import ElementTree as ET

W_NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


class DocxError(Exception):
    pass


def is_owner_file(path: str) -> bool:
    """Word lock file: '~$' name prefix and not a ZIP container."""
    if not os.path.basename(path).startswith('~$'):
        return False
    try:
        with open(path, 'rb') as fh:
            return fh.read(4) != b'PK\x03\x04'
    except OSError:
        return True


def extract_text(path: str) -> str:
    """Paragraph text of word/document.xml, one paragraph per line.

    Tabs and explicit breaks inside a paragraph become spaces/newlines; runs
    are concatenated verbatim. Nothing is normalised here - cleaning happens
    downstream exactly as for InPage text.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read('word/document.xml')
    except (zipfile.BadZipFile, KeyError, OSError) as e:
        raise DocxError('%s: %s' % (type(e).__name__, e))
    root = ET.fromstring(xml)
    lines = []
    for para in root.iter(W_NS + 'p'):
        parts = []
        for node in para.iter():
            if node.tag == W_NS + 't' and node.text:
                parts.append(node.text)
            elif node.tag == W_NS + 'tab':
                parts.append(' ')
            elif node.tag in (W_NS + 'br', W_NS + 'cr'):
                parts.append('\n')
        text = re.sub(r'[ \t]+', ' ', ''.join(parts)).strip()
        if text:
            lines.append(text)
    return '\n'.join(lines)
