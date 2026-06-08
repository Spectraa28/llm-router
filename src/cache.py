import os 
import json
import logging 
import numpy as np 
import faiss
import redis 
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.88"))
CACHE_STORE_DIR = os.getenv("CACHE_STORE_DIR", "cache_store")
KEY_PREFIX = "llmrouter:cache"
MODEL_CACHE_DIR = os.getenv("SENTENCE_TRANSFORMERS_HOME", "/app/model_cache")

FAISS_INDEX_PATH = os.path.join(CACHE_STORE_DIR, "faiss.index")
META_PATH = os.path.join(CACHE_STORE_DIR, "meta.json")


class SemanticCache:
    def __init__(self):
        self.threshold = CACHE_SIMILARITY_THRESHOLD
        self._encoder = None  # lazy — not loaded yet
        self._dim = 384       # all-MiniLM-L6-v2 is always 384, hardcode avoids early load

        # ── Redis ─────────────────────────────────────────────────────────────
        try:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                password=os.getenv("REDIS_PASSWORD", None),
                decode_responses=True,
                socket_connect_timeout=2
            )
            self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis connection established.")
        except Exception as e:
            logger.warning(f"Redis unavailable — cache writes disabled. Reason: {e}")
            self.redis_available = False

        # ── FAISS ─────────────────────────────────────────────────────────────
        os.makedirs(CACHE_STORE_DIR, exist_ok=True)

        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(META_PATH):
            logger.info("Loading persistent FAISS index from disk.")
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(META_PATH, "r") as f:
                meta = json.load(f)
            self.next_id = meta.get("next_id", 0)
            logger.info(f"Cache restored — {self.index.ntotal} entries, next_id={self.next_id}")

            if self.redis_available:
                llmrouter_keys = len(self.redis_client.keys(f"{KEY_PREFIX}:*"))
                if llmrouter_keys < self.index.ntotal:
                    logger.warning(
                        f"Redis has {llmrouter_keys} llmrouter keys but FAISS has "
                        f"{self.index.ntotal} entries — out of sync. Resetting both."
                    )
                    self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self._dim))
                    self.next_id = 0
                    self._save()
                else:
                    logger.info(f"Redis and FAISS in sync — {llmrouter_keys} keys.")
        else:
            logger.info("No persistent index found — starting fresh.")
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self._dim))
            self.next_id = 0

    # ── Lazy encoder ──────────────────────────────────────────────────────────
    @property
    def encoder(self):
        if self._encoder is None:
            logger.info("Loading SentenceTransformer model (first request)...")
            self._encoder = SentenceTransformer(
                "all-MiniLM-L6-v2",
                cache_folder=MODEL_CACHE_DIR
            )
            logger.info("Model loaded.")
        return self._encoder

    # ── Persistence ───────────────────────────────────────────────────────────
    def _save(self):
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(META_PATH, "w") as f:
            json.dump({"next_id": self.next_id}, f)
        logger.debug("FAISS index and metadata saved to disk.")

    # ── Get ───────────────────────────────────────────────────────────────────
    def get(self, query: str) -> str | None:
        if self.index.ntotal == 0:
            return None
        if not self.redis_available:
            logger.warning("Cache get skipped — Redis unavailable.")
            return None

        clean_query = query.strip().lower()
        query_vector = self.encoder.encode(
            [clean_query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_vector, k=1)
        best_score = float(scores[0][0])
        best_id = int(indices[0][0])

        logger.info(f"Cache lookup | score={best_score:.4f} | id={best_id} | threshold={self.threshold}")

        if best_score >= self.threshold:
            try:
                cached = self.redis_client.get(f"{KEY_PREFIX}:{best_id}")
                if cached:
                    logger.info(f"Cache HIT | score={best_score:.4f} | id={best_id}")
                    return cached
                else:
                    logger.warning(f"Cache MISS (FAISS hit but Redis key missing) | id={best_id}")
                    return None
            except Exception as e:
                logger.error(f"Redis GET failed: {e}")
                return None

        return None

    # ── Set ───────────────────────────────────────────────────────────────────
    def set(self, query: str, answer: str) -> None:
        if not self.redis_available:
            logger.warning("Cache write skipped — Redis unavailable.")
            return

        clean_query = query.strip().lower()
        query_vector = self.encoder.encode(
            [clean_query], normalize_embeddings=True
        ).astype(np.float32)

        try:
            self.redis_client.set(f"{KEY_PREFIX}:{self.next_id}", answer)
        except Exception as e:
            logger.error(f"Redis SET failed — skipping cache write: {e}")
            return

        vector_id = np.array([self.next_id], dtype=np.int64)
        self.index.add_with_ids(query_vector, vector_id)
        self.next_id += 1
        self._save()

        logger.info(f"Cache SET | id={self.next_id - 1} | total_entries={self.index.ntotal}")

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def cache_size(self) -> int:
        return self.index.ntotal