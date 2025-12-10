# app.py
import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv
from rag import MiniRAG

load_dotenv()
TG_TOKEN = os.getenv("TG_TOKEN")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

user_history = {}

try:
    rag = MiniRAG()
except Exception as e:
    print("Error initializing MiniRAG:", e)
    raise

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Mini-RAG (Ollama) bot ready. Use /ask <query>.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = "/ask <query> — query the knowledge base\n/help — show commands\n/summarize — last 3 queries"
    await update.message.reply_text(msg)

async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args).strip()
    if not query:
        await update.message.reply_text("Usage: /ask <your question>")
        return

    uid = str(update.message.from_user.id)
    user_history.setdefault(uid, []).append(("user", query))
    if len(user_history[uid]) > 10:
        user_history[uid] = user_history[uid][-10:]

    try:
        answer = rag.ask(query)
    except Exception as e:
        logger.exception("RAG error: %s", e)
        answer = "Sorry — an internal error occurred."

    user_history[uid].append(("bot", answer))
    if len(user_history[uid]) > 10:
        user_history[uid] = user_history[uid][-10:]

    await update.message.reply_text(answer)

async def summarize_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.message.from_user.id)
    history = user_history.get(uid, [])
    user_msgs = [t for who,t in history if who == "user"][-3:]
    if not user_msgs:
        await update.message.reply_text("No recent queries found.")
        return
    await update.message.reply_text("Your last queries:\n- " + "\n- ".join(user_msgs))

def main():
    if not TG_TOKEN:
        print("❌ You must set TG_TOKEN in .env file")
        return

    app = ApplicationBuilder().token(TG_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("summarize", summarize_cmd))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
