from __future__ import annotations

from typing import Any

def compact_names(users: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for user in users:
        name = user.get("name") or user.get("nickname")
        if name:
            names.append(name)
    return names


def flatten_message(message: dict[str, Any]) -> dict[str, Any]:
    author = message.get("author") or {}
    reference = message.get("reference") or {}

    flat: dict[str, Any] = {
        "post_id": message.get("id"),
        "user_name": author.get("name"),
        "timestamp": message.get("timestamp"),
        "type": message.get("type"),
        "content": message.get("content") or "",
    }

    if author.get("id"):
        flat["user_id"] = author["id"]

    if message.get("timestampEdited"):
        flat["edited_at"] = message["timestampEdited"]

    if reference.get("messageId"):
        flat["reply_to_post_id"] = reference["messageId"]

    attachments = message.get("attachments") or []
    if attachments:
        flat["attachments"] = [
            {
                "id": attachment.get("id"),
                "file_name": attachment.get("fileName"),
                "url": attachment.get("url"),
            }
            for attachment in attachments
        ]

    mentions = compact_names(message.get("mentions") or [])
    if mentions:
        flat["mentions"] = mentions

    return flat


def flatten_export(export: dict[str, Any]) -> list[dict[str, Any]]:
    return [flatten_message(message) for message in export.get("messages", [])]
