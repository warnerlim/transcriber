# text_utils.py
import hashlib
import re

def preprocess_wechat_text(text: str) -> str:
    if not text:
        return ""

    text = "".join(
        ch for ch in text
        if ch in "\t\n\r" or (ord(ch) >= 0x20 and not 0x7F <= ord(ch) <= 0x9F)
    )
    text = re.sub(r"[ ​﻿  -   　]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate_message_id(
    chat_id: str,
    sender: str,
    receiver: str,
    text: str,
    timestamp: int,
) -> str:
    raw = f"{chat_id}|{sender}|{receiver}|{text}|{timestamp}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_pdf_chunk_id(
    file_name: str,
    page_index: int,
    chunk_index: int,
    chunk: str,
) -> str:
    raw = f"{file_name}|page_{page_index + 1}|chunk_{chunk_index}|{chunk[:50]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()