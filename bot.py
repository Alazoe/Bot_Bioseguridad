"""
from __future__ import annotations

# bot.py - Bot de Telegram para consultas de bioseguridad.

Uso:
    python bot.py

Comandos disponibles en Telegram:
    /start   - Mensaje de bienvenida y estado del sistema
    /estado  - Muestra documentos indexados en la base de conocimiento
    /limpiar - Borra el historial de conversación del chat actual
    /ayuda   - Instrucciones de uso
"""

import logging
import os

from dotenv import load_dotenv
from telegram import BotCommand, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from rag import answer_question, get_kb_stats

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Historial de conversación por chat_id: {chat_id: [{"role": ..., "content": ...}]}
conversation_histories: dict[int, list[dict]] = {}
MAX_HISTORY = 10  # Máximo de mensajes a mantener por chat (5 intercambios)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kb_summary() -> str:
    stats = get_kb_stats()
    if stats["count"] == 0:
        return (
            "Base de conocimiento: *vacía*\n"
            "Ejecuta `python ingest.py` para indexar documentos."
        )
    sources = "\n".join(f"  • {s}" for s in stats["sources"])
    return (
        f"Base de conocimiento: *{stats['count']} fragmentos indexados*\n"
        f"Documentos cargados:\n{sources}"
    )


def _escape(text: str) -> str:
    """Escapa caracteres especiales para MarkdownV2."""
    special = r"\_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{c}" if c in special else c for c in text)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb_info = _kb_summary()
    text = (
        "Hola\\! Soy el *Bot de Bioseguridad* 🧬\n\n"
        "Puedo responder preguntas sobre bioseguridad basándome en documentos "
        "técnicos, reglamentos del SAG, normativas OIE y literatura especializada\\.\n\n"
        f"{_escape(kb_info)}\n\n"
        "*Comandos disponibles:*\n"
        "/estado \\- Estado de la base de conocimiento\n"
        "/limpiar \\- Reiniciar historial de conversación\n"
        "/ayuda \\- Cómo usar el bot\n\n"
        "Escribe tu pregunta directamente para comenzar\\."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_estado(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = get_kb_stats()
    if stats["count"] == 0:
        msg = (
            "La base de conocimiento está *vacía*\\.\n\n"
            "Para agregar documentos:\n"
            "1\\. Copia tus PDFs en la carpeta `documents/`\n"
            "2\\. Ejecuta `python ingest.py` en la terminal"
        )
    else:
        sources_list = "\n".join(f"  • {_escape(s)}" for s in stats["sources"])
        msg = (
            f"*Base de conocimiento*\n\n"
            f"Fragmentos indexados: *{stats['count']}*\n\n"
            f"Documentos:\n{sources_list}"
        )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_limpiar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    conversation_histories.pop(chat_id, None)
    await update.message.reply_text(
        "Historial de conversación reiniciado\\. Empecemos de nuevo\\.",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def cmd_ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "*Cómo usar el Bot de Bioseguridad*\n\n"
        "*Hacer preguntas:*\n"
        "Escribe directamente tu consulta\\. Por ejemplo:\n"
        "• _¿Cuáles son los requisitos de bioseguridad para importar bovinos?_\n"
        "• _¿Qué medidas de desinfección exige el SAG para predios avícolas?_\n"
        "• _¿Cómo se clasifica un agente de riesgo biológico nivel 2?_\n\n"
        "*Comandos:*\n"
        "/start \\- Bienvenida y estado\n"
        "/estado \\- Ver documentos cargados\n"
        "/limpiar \\- Borrar historial del chat\n"
        "/ayuda \\- Esta ayuda\n\n"
        "*Agregar documentos:*\n"
        "Copia PDFs en la carpeta `documents/` y ejecuta:\n"
        "`python ingest\\.py`"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    question = update.message.text.strip()

    if not question:
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    history = conversation_histories.get(chat_id, [])

    try:
        answer = answer_question(question, history)

        # Actualizar historial (guardamos el texto original, no el aumentado con contexto)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        conversation_histories[chat_id] = history[-MAX_HISTORY:]

        # Telegram tiene límite de 4096 caracteres por mensaje
        if len(answer) > 4000:
            for i in range(0, len(answer), 4000):
                await update.message.reply_text(answer[i : i + 4000])
        else:
            await update.message.reply_text(answer)

    except Exception as e:
        logger.error(f"Error al procesar mensaje de chat {chat_id}: {e}", exc_info=True)
        await update.message.reply_text(
            "Ocurrió un error al procesar tu consulta. Por favor intenta de nuevo.\n"
            "Si el problema persiste, verifica que ANTHROPIC_API_KEY esté configurada en .env"
        )


# ---------------------------------------------------------------------------
# Setup y arranque
# ---------------------------------------------------------------------------

async def post_init(application: Application):
    await application.bot.set_my_commands(
        [
            BotCommand("start", "Iniciar el bot"),
            BotCommand("estado", "Ver documentos indexados"),
            BotCommand("limpiar", "Reiniciar historial de conversación"),
            BotCommand("ayuda", "Instrucciones de uso"),
        ]
    )


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN no está configurado.\n"
            "Crea un archivo .env con: TELEGRAM_BOT_TOKEN=tu_token_aqui"
        )

    app = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("estado", cmd_estado))
    app.add_handler(CommandHandler("limpiar", cmd_limpiar))
    app.add_handler(CommandHandler("ayuda", cmd_ayuda))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot de Bioseguridad iniciado. Presiona Ctrl+C para detener.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
