import os 
import json
import logging 
import numpy as np 
import faiss
import redis 
from sentence_transformers import SentenceTransformer
from  dotenv  import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD","0.88"))
CACHE_STORE_DIR = os.getenv("CACHE_STORE_DIR","cache_store")

FAISS_INDEX_PATH = os.path.join(CACHE_STORE_DIR, "faiss.index")
META_PATH = os.path.join(CACHE_STORE_DIR, "meta.json")

class SemanticCache:
    def __init__(self):
        self.threshold = CACHE_SIMILARITY_THRESHOLD
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        dim = self.encoder.get_embedding_dimension()
        # REDIS 
        try:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST","localhost")
                ,port=int(os.getenv("REDIS_PORT","6379")),
                decode_responses=True,
                socket_connect_timeout=2
            )
            
            self.redis_client.ping()
            self.redis_available  = True
            logger.info("Reddis Connection established")
        except Exception as e :
            logger.warning(f"Redis unvailable - cache writes disabled .  Reason : {e}")
            self.redis_available = False
            
            
        # FAISS 
        os.makedirs(CACHE_STORE_DIR,exist_ok=True)
        
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(META_PATH):
            logger.info("Loading persistent Faiss index from disk. ")
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(META_PATH, "r") as f:
                meta = json.load(f)
            self.next_id = meta.get("next_id",0 )
            logger.info(f"Cache restored   -- {self.index.ntotal} entries , next_id = {self.next_id} ")
            
        else:
            logger.info("No persistent index found - Starting fresh")
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
            self.next_id = 0
            
    # PErsistence 
    def _save(self):
        """
        Persistent FAISS index and metadata to disk 
        """
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(META_PATH, "w") as f:
            json.dump({"next_id": self.next_id}, f)
        logger.debug("FAISS index and metadata saved to disk")
        
    # GET 
    def get(self,query:str) -> str | None:
        if self.index.ntotal == 0 or not self.redis_available:
            return None
        
        clean_query = query.strip().lower()
        query_vector = self.encoder.encode(
            [clean_query], normalize_embeddings=True
        ).astype(np.float32)
        
        
        scores, indices = self.index.search(query_vector, k=1)
        
        best_score = float(scores[0][0])
        best_id = float(indices[0][0])
        
        logger.debug(f"Cache lookup | query='{query} | score = {best_score:.4f} | id = {best_id} ")
        
        if best_score >= self.threshold:
            try:
                cached = self.redis_client.get(f"cache: {best_id}")
                if cached:
                    logger.info(f"Cache Hit | score={best_id:.4f}")
                    return cached
                logger.warning(f"Cache MISS (FAISS hit but redis key missing) | id={best_id}")
            except Exception as e:
                logger.error(f"Redis GET failed after startup -  cache degraded: {e}")
                self.redis_available = False
        return None
    
    # SET 
    def set(self,query:str, answer:str) -> None:
        if not self.redis_available:
            logger.warning("Cache  write skipped -  Redis unavailable")
            return
        
        clean_query = query.strip().lower()
        query_vector = self.encoder.encode(
            [clean_query],normalize_embeddings=True
        ).astype(np.float32)
        
        # Writitng redis firsst 
        try:
            self.redis_client.set(f"cache:{self.next_id}", answer)
        except Exception as e:
            logger.error(f"Redis SET failed — skipping cache write: {e}")
            return
        
        #WRitinng FAISS second
        vector_id = np.array([self.next_id],dtype=np.int64)
        self.index.add_with_ids(query_vector,vector_id)
        
        
        self.next_id += 1
        
        # Persist to disk after every write 
        self._save()
        
        logger.info(f"Cache SET | id= {self.next_id - 1} | total_enteries={self.index.ntotal}")
        
        @property
        def cache_size(self) -> int:
            return self.index.ntotal