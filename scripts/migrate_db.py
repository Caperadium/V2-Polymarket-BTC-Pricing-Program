
import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DB_DIR = Path(__file__).parent.parent
DB_PATH = DB_DIR / "polymarket_console.db"

def migrate():
    if not DB_PATH.exists():
        logger.info("Database not found, skipping migration.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    try:
        cursor = conn.cursor()
        
        # Check if intent_key column exists
        cursor.execute("PRAGMA table_info(intents)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "intent_key" not in columns:
            logger.info("Adding intent_key column to intents table...")
            cursor.execute("ALTER TABLE intents ADD COLUMN intent_key TEXT")
            # We should also add an index for performance and potential uniqueness checks
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_intents_key ON intents(intent_key)")
            conn.commit()
            logger.info("Migration successful.")
        else:
            logger.info("intent_key column already exists.")
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
