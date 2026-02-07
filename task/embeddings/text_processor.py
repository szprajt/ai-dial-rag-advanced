from enum import StrEnum
import os
import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def _truncate_table(self):
        """Truncate the vectors table"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE vectors RESTART IDENTITY;")
            conn.commit()
        finally:
            conn.close()

    def _save_chunk(self, document_name: str, text: str, embedding: list[float]):
        """Save a single chunk with its embedding to the database"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
                    (document_name, text, str(embedding))
                )
            conn.commit()
        finally:
            conn.close()

    def process_text_file(self, file_path: str, chunk_size: int = 300, overlap: int = 40, truncate: bool = False):
        """
        Process a text file: chunk it, generate embeddings, and store in DB.
        """
        if truncate:
            self._truncate_table()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        chunks = chunk_text(content, chunk_size, overlap)
        
        if not chunks:
            return

        # Generate embeddings for all chunks
        # Note: Depending on API limits, we might need to batch this if chunks list is very large.
        # Assuming reasonable file size for this task.
        embeddings_map = self.embeddings_client.get_embeddings(chunks)

        document_name = os.path.basename(file_path)
        
        for i, chunk in enumerate(chunks):
            if i in embeddings_map:
                self._save_chunk(document_name, chunk, embeddings_map[i])

    def search(self, query: str, mode: SearchMode = SearchMode.COSINE_DISTANCE, top_k: int = 5, min_score: float = 0.5) -> list[dict]:
        """
        Search for relevant context in the database based on the query.
        """
        # Generate embedding for the query
        query_embeddings_map = self.embeddings_client.get_embeddings([query])
        if not query_embeddings_map:
            return []
        
        query_embedding = query_embeddings_map[0]
        embedding_str = str(query_embedding)

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if mode == SearchMode.EUCLIDIAN_DISTANCE:
                    # Euclidean distance: smaller is better. 
                    # We want distance < threshold? Or convert to similarity?
                    # Usually for RAG we want "closest".
                    # The requirement says "min_score". 
                    # For cosine distance (1 - cosine_similarity), range is 0 to 2. 0 is identical.
                    # For cosine similarity, range is -1 to 1. 1 is identical.
                    # pgvector <=> operator returns cosine distance (1 - cosine similarity).
                    # So smaller is better.
                    # If min_score is similarity threshold (e.g. 0.7), then distance should be <= (1 - 0.7) = 0.3.
                    
                    # Let's assume min_score is a similarity score (0 to 1) where 1 is best match.
                    # For cosine distance (<=>), distance = 1 - similarity. So distance <= 1 - min_score.
                    
                    # For Euclidean distance (<->), it's unbounded (0 to infinity). 
                    # Hard to map min_score directly without normalization.
                    # However, let's stick to the operator.
                    
                    # Let's assume for this task, we just order by distance and limit.
                    # And maybe filter if distance is too large?
                    # The prompt mentions "min_score: Similarity threshold (range: 0.1-0.99, default: 0.5)".
                    # This strongly suggests Cosine Similarity.
                    
                    # If mode is Euclidean, we use <->.
                    sql = f"""
                        SELECT id, document_name, text, (embedding <-> %s::vector) as distance
                        FROM vectors
                        ORDER BY distance ASC
                        LIMIT %s
                    """
                    cur.execute(sql, (embedding_str, top_k))
                    
                else: # COSINE_DISTANCE
                    # <=> is cosine distance.
                    # We want to filter where similarity >= min_score.
                    # similarity = 1 - distance.
                    # 1 - distance >= min_score  =>  distance <= 1 - min_score.
                    
                    max_distance = 1.0 - min_score
                    
                    sql = f"""
                        SELECT id, document_name, text, (embedding <=> %s::vector) as distance
                        FROM vectors
                        WHERE (embedding <=> %s::vector) <= %s
                        ORDER BY distance ASC
                        LIMIT %s
                    """
                    cur.execute(sql, (embedding_str, embedding_str, max_distance, top_k))

                results = cur.fetchall()
                return results
        finally:
            conn.close()
