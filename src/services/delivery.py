"""
Delivery channels for item summaries (digest).

Supported: Email (SMTP, HTML), Telegram (Bot API, Markdown).
Default: Send all news to both email and Telegram (per-persona selection can be added later).
Fallback: File output (JSON + Markdown) is always generated.
"""

import os
import json
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from urllib.parse import quote

try:
    from dotenv import load_dotenv
    load_dotenv()
    DB_PATH=Path(os.getenv("DB_PATH"))
except ImportError:
    pass

# Import digest entries from summarization (TextRank or LLM)
from ..tools.summarization_textrank import get_digest_entries, SUMMARY_SOURCE

DEFAULT_SOURCE = SUMMARY_SOURCE


def _env(key, default=None):
    return os.environ.get(key, default)


def get_delivery_config(persona="GENAI_NEWS"):
    """
    Read delivery configuration from environment.
    Default: email + telegram (all news to both). Override with DELIVERY_CHANNELS if needed.
    Per-persona recipient overrides (for later): {PERSONA}_EMAIL_TO, {PERSONA}_TELEGRAM_CHAT_ID.
    """
    base = persona.upper().replace("-", "_")
    # Default: send to both email and telegram; file is always included
    raw = _env(f"{base}_DELIVERY") or _env("DELIVERY_CHANNELS") or "email,telegram"
    channels = [c.strip().lower() for c in raw.split(",") if c.strip()]
    if "file" not in channels:
        channels.append("file")  # fallback always

    email_to = _env(f"{base}_EMAIL_TO") or _env("EMAIL_TO") or ""
    telegram_chat_id = _env(f"{base}_TELEGRAM_CHAT_ID") or _env("TELEGRAM_CHAT_ID") or ""

    return {
        "channels": channels,
        "email_to": email_to,
        "telegram_chat_id": telegram_chat_id,
        "smtp_host": _env("SMTP_HOST", "localhost"),
        "smtp_port": int(_env("SMTP_PORT", "587")),
        "smtp_user": _env("SMTP_USER", ""),
        "smtp_password": _env("SMTP_PASSWORD", ""),
        "smtp_from": _env("SMTP_FROM") or _env("SMTP_USER", "digest@local"),
        "smtp_use_tls": _env("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes"),
        "telegram_bot_token": _env("TELEGRAM_BOT_TOKEN", ""),
        "digest_subject_prefix": _env("DIGEST_SUBJECT_PREFIX", "Daily Digest"),
    }


def build_digest_html(entries, title="Daily Digest"):
    """Build HTML body for email from digest entries."""
    if not entries:
        return f"<p>No items in this digest.</p>"

    parts = [f"<h1>{title}</h1>", "<p>Item summaries:</p>", "<ul>"]
    for e in entries:
        headline = _escape_html(e.get("headline") or "(No title)")
        lead = _escape_html(e.get("lead") or "")
        why = _escape_html(e.get("why_it_matters") or "")
        audience = _escape_html(e.get("target_audience") or "developer")
        topic = _escape_html(e.get("topic") or "")
        likes = e.get("likes", 0)
        comments = e.get("comments", 0)
        url = (e.get("url") or "").strip()
        url_attr = f' href="{_escape_attr(url)}"' if url else ""
        link_tag = f'<a{url_attr}>{headline}</a>' if url else headline
        parts.append("<li>")
        parts.append(f"<strong>{link_tag}</strong>")
        if topic:
            parts.append(f" <em>[{topic}]</em>")
        parts.append(f" &mdash; {audience}")
        parts.append(f" | 👍 {likes} | 💬 {comments}")
        if lead:
            parts.append(f"<p>{lead}</p>")
        if why:
            parts.append(f"<p><small>Why it matters: {why}</small></p>")
        parts.append("</li>")
    parts.append("</ul>")
    return "\n".join(parts)


def _escape_html(s):
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _escape_attr(s):
    return s.replace("&", "&amp;").replace('"', "&quot;")


def _escape_telegram_markdown(s):
    """
    Escape text for Telegram MarkdownV2.
    All special characters must be escaped with backslash.
    """
    if s is None:
        return ""
    s = str(s)
    # Characters that need escaping in MarkdownV2
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    for ch in escape_chars:
        s = s.replace(ch, "\\" + ch)
    # Backslash must be escaped last (or handle separately)
    s = s.replace("\\\\", "\\")  # Avoid double-escaping
    return s


def _build_entry_markdown(entry, escape_for_telegram=True):
    """
    Build markdown for a single entry.
    Returns the formatted entry text.
    """
    def esc(t):
        return _escape_telegram_markdown(t) if escape_for_telegram else (t or "")
    
    headline = entry.get("headline") or "(No title)"
    url = (entry.get("url") or "").strip()
    lead = entry.get("lead") or ""
    why = entry.get("why_it_matters") or ""
    audience = entry.get("target_audience") or "developer"
    topic = entry.get("topic") or ""
    likes = entry.get("likes", 0)
    comments = entry.get("comments", 0)
    
    entry_lines = []
    
    if escape_for_telegram:
        headline_esc = esc(headline)
        lead_esc = esc(lead)
        why_esc = esc(why)
        audience_esc = esc(audience)
        topic_esc = esc(topic)
    else:
        headline_esc = headline
        lead_esc = lead
        why_esc = why
        audience_esc = audience
        topic_esc = topic
    
    # Build the link
    if url:
        entry_lines.append(f"• [{headline_esc}]({url})")
    else:
        entry_lines.append(f"• *{headline_esc}*")
    
    # Topic and audience
    if topic_esc:
        entry_lines.append(f"   _{topic_esc}_ — {audience_esc}")
    else:
        entry_lines.append(f"   {audience_esc}")
    
    # Engagement
    if escape_for_telegram:
        entry_lines.append(f"   👍 {likes} \\| 💬 {comments}")
    else:
        entry_lines.append(f"   👍 {likes} \\| 💬 {comments}")
    
    # Add lead
    if lead_esc:
        entry_lines.append(f"   {lead_esc}")
    
    # Add why it matters
    if why_esc:
        if escape_for_telegram:
            entry_lines.append(f"*Why it matters:* {why_esc}")
        else:
            entry_lines.append(f"*Why it matters*: {why_esc}")
    
    return "\n".join(entry_lines)


def build_digest_messages(entries, title="Daily Digest", escape_for_telegram=True, max_message_length=4000):
    """
    Build multiple Telegram messages ensuring each entry stays intact within a message.
    Returns list of message texts.
    If an entry doesn't fit in current message, it goes to the next message.
    All entries are guaranteed to be included.
    """
    if not entries:
        no_items = "No items in this digest\\." if escape_for_telegram else "No items in this digest."
        title_safe = _escape_telegram_markdown(title) if escape_for_telegram else title
        return [f"*{title_safe}*\n\n{no_items}"]
    
    messages = []
    title_safe = _escape_telegram_markdown(title) if escape_for_telegram else title
    header = f"*{title_safe}*\n\n"
    
    current_message = header
    current_length = len(header)
    buffer_for_safety = 150  # Safety buffer for Telegram limits
    
    for i, entry in enumerate(entries, 1):
        # Build the entry
        entry_text = _build_entry_markdown(entry, escape_for_telegram)
        entry_length = len(entry_text) + 2  # +2 for newlines
        entry_with_separator = entry_text + "\n\n"
        
        # Check if entry fits in current message
        if current_length + entry_length + buffer_for_safety >= max_message_length:
            # Current message is full, save it and start a new one
            messages.append(current_message.strip())
            current_message = header
            current_length = len(header)
        
        # Add entry to current message
        current_message += entry_with_separator
        current_length += len(entry_with_separator)
    
    # Add the last message
    if current_message.strip() != header.strip():
        messages.append(current_message.strip())
    
    return messages


def build_digest_markdown(entries, title="Daily Digest", escape_for_telegram=True, max_message_length=999999):
    """
    Build full Markdown body for file output (no message limits).
    When escape_for_telegram=True, escapes special chars for Telegram MarkdownV2.
    """
    if not entries:
        no_items = "No items in this digest\\." if escape_for_telegram else "No items in this digest."
        title_safe = _escape_telegram_markdown(title) if escape_for_telegram else title
        return f"*{title_safe}*\n\n{no_items}"

    title_safe = _escape_telegram_markdown(title) if escape_for_telegram else title
    lines = [f"*{title_safe}*", ""]
    
    for i, e in enumerate(entries, 1):
        entry_text = _build_entry_markdown(e, escape_for_telegram)
        lines.append(entry_text)
        lines.append("")
    
    return "\n".join(lines).strip()


def write_fallback_files(entries, output_dir="output", title="Daily Digest", source=None):
    """
    Always write digest to JSON and Markdown files (fallback).
    Returns paths: (json_path, md_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")
    base_name = f"digest_{ts}"
    if source:
        base_name = f"digest_{source}_{ts}"
    json_path = os.path.join(output_dir, f"{base_name}.json")
    md_path = os.path.join(output_dir, f"{base_name}.md")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "title": title,
        "total_items": len(entries),
        "entries": entries,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Write markdown without Telegram escaping for file output (full content)
    md_content = build_digest_markdown(entries, title=title, escape_for_telegram=False, max_message_length=999999)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return json_path, md_path


def send_email(html_content, subject, to_address, config, verbose=True):
    """Send HTML email via SMTP."""
    if not to_address or not to_address.strip():
        if verbose:
            print("[Delivery] Skipping email: no EMAIL_TO or PERSONA_EMAIL_TO set.")
        return False
    host = config["smtp_host"]
    port = config["smtp_port"]
    user = config["smtp_user"]
    password = config["smtp_password"]
    from_addr = config["smtp_from"]
    use_tls = config["smtp_use_tls"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_address.strip()
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        with smtplib.SMTP(host, port) as s:
            if use_tls:
                s.starttls()
            if user and password:
                s.login(user, password)
            s.sendmail(from_addr, [a.strip() for a in to_address.split(",")], msg.as_string())
        if verbose:
            print(f"[Delivery] Email sent to {to_address.strip()}")
        return True
    except Exception as e:
        if verbose:
            print(f"[Delivery] Email failed: {e}")
        return False


async def _send_telegram_async(messages, chat_id, bot_token, verbose=True):
    """Send multiple messages via python-telegram-bot (Bot API). Uses MarkdownV2 parse_mode."""
    from telegram import Bot

    bot = Bot(token=bot_token.strip())
    chat_id_str = str(chat_id).strip()

    # Send each message
    for i, message_text in enumerate(messages, 1):
        await bot.send_message(
            chat_id=chat_id_str,
            text=message_text,
            parse_mode="MarkdownV2",
            disable_web_page_preview=True,
        )
        if verbose and len(messages) > 1:
            print(f"[Delivery] Telegram message {i}/{len(messages)} sent ({len(message_text)} chars)")
        elif verbose:
            print(f"[Delivery] Telegram message sent ({len(message_text)} chars)")


def send_telegram(messages, chat_id, bot_token, verbose=True):
    """
    Send messages via Telegram Bot API (python-telegram-bot).
    Uses MarkdownV2 parse_mode. Sends multiple messages if needed, ensuring no entry is split.
    messages: list of message texts or single string
    """
    if not bot_token or not chat_id or not str(chat_id).strip():
        if verbose:
            print("[Delivery] Skipping Telegram: no TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID set.")
        return False
    
    # Convert single string to list
    if isinstance(messages, str):
        messages = [messages]
    
    try:
        import asyncio
        asyncio.run(_send_telegram_async(messages, chat_id, bot_token, verbose=verbose))
        return True
    except Exception as e:
        if verbose:
            print(f"[Delivery] Telegram failed: {e}")
        return False


def run_delivery(
    persona="GENAI_NEWS",
    db_path="mydb.db",
    source=None,
    output_dir="output",
    verbose=True,
):
    """
    Load digest entries, write JSON + Markdown files (always), then send via
    channels configured for this persona (email, telegram).
    Entries are sorted by total engagement (likes + comments) in descending order.
    """
    source = source or DEFAULT_SOURCE
    try:
        entries = get_digest_entries(db_path=DB_PATH, source=source)
    except Exception as e:
        if verbose:
            print(f"[Delivery] Failed to load digest entries: {e}")
        entries = []

    # Sort entries by total engagement (likes + comments) in descending order
    entries.sort(key=lambda e: e.get("likes", 0) + e.get("comments", 0), reverse=True)

    config = get_delivery_config(persona=persona)
    title = f"{config['digest_subject_prefix']} ({persona})"

    # 1. Fallback: always write JSON + Markdown (full content)
    json_path, md_path = write_fallback_files(
        entries, output_dir=output_dir, title=title, source=source
    )
    if verbose:
        print(f"[Delivery] Fallback files: {json_path}, {md_path}")

    channels = config["channels"]
    html = build_digest_html(entries, title=title)
    md_telegram_messages = build_digest_messages(entries, title=title, escape_for_telegram=True)
    subject = f"{title} — {len(entries)} items"

    # 2. Email (HTML)
    if "email" in channels:
        send_email(
            html,
            subject=subject,
            to_address=config["email_to"],
            config=config,
            verbose=verbose,
        )

    # 3. Telegram (Multiple messages if needed, ensuring entries stay intact)
    if "telegram" in channels:
        send_telegram(
            md_telegram_messages,
            chat_id=config["telegram_chat_id"],
            bot_token=config["telegram_bot_token"],
            verbose=verbose,
        )

    return {"entries": len(entries), "json_path": json_path, "md_path": md_path}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Deliver digest via email, Telegram, and file fallback")
    p.add_argument("--persona", default="GENAI_NEWS", help="Persona for channel config (e.g. GENAI_NEWS_DELIVERY)")
    p.add_argument("--db", default="mydb.db", help="Database path")
    p.add_argument("--source", default=None, help="item_summaries source (default: TEXTRANK)")
    p.add_argument("--output-dir", default="output", help="Directory for JSON/MD fallback files")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    run_delivery(
        persona=args.persona,
        db_path=args.db,
        source=args.source,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )