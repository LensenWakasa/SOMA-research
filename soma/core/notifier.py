"""
acsis/interface/notifier.py
============================
Sends Acsis discoveries and results to Lensen + Simon via Telegram.

Setup:
  1. Create a bot at @BotFather → get token
  2. Get your chat_id: message @userinfobot
  3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars
"""
from __future__ import annotations
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)


class Notifier:
    """
    Sends messages to Telegram.
    Falls back to console print if not configured.

    Usage:
        n = Notifier(cfg)
        await n.send("Acsis has discovered something interesting...")
    """

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.token = getattr(cfg, 'telegram_token', '') if cfg else ''
        self.chat_id = getattr(cfg, 'telegram_chat_id', '') if cfg else ''
        self.enabled = bool(self.token and self.chat_id)
        if self.enabled:
            logger.info("[NOTIFY] Telegram notifications enabled")
        else:
            logger.info("[NOTIFY] Telegram not configured — printing to console only")

    async def send(self, message: str, parse_mode: str = "HTML"):
        """Send a message. Auto-truncates to Telegram's 4096 char limit."""
        if len(message) > 4000:
            message = message[:3900] + "\n\n[...truncated]"

        if self.enabled:
            await self._send_telegram(message, parse_mode)
        else:
            # Console output with visual separator
            print("\n" + "═"*60)
            print("ACSIS UPDATE")
            print("═"*60)
            print(message)
            print("═"*60 + "\n")

    async def send_discovery(self, discovery: str, question: str = ""):
        """Formatted discovery notification."""
        text = (
            f"<b>★ ACSIS DISCOVERY</b>\n\n"
            f"<b>Question:</b> {question}\n\n"
            f"<b>Finding:</b> {discovery}\n\n"
            f"<i>Review and validate before citing.</i>"
        )
        await self.send(text)

    async def send_summary(self, n_questions: int, n_discoveries: int, n_facts: int):
        """Daily/session summary."""
        text = (
            f"<b>ACSIS SESSION SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"Questions answered: {n_questions}\n"
            f"Discoveries flagged: {n_discoveries}\n"
            f"Facts stored: {n_facts}\n"
        )
        await self.send(text)

    async def _send_telegram(self, text: str, parse_mode: str = "HTML"):
        try:
            import aiohttp
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        logger.info("[NOTIFY] Message sent")
                    else:
                        data = await resp.text()
                        logger.error(f"[NOTIFY] Telegram error {resp.status}: {data}")
        except Exception as e:
            logger.error(f"[NOTIFY] Failed to send: {e}")
            # Fall back to console
            print(text)
