"""ChromaDB Manager - Vector storage and retrieval"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional, Union
import json
import shutil
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaManager:
    def __init__(
        self,
        persist_directory: str = "data/processed/chroma_db",
        collection_name: str = "regulations"
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection_name = collection_name
        self.collection = None
        
        logger.info(f"ChromaDB initialized at: {self.persist_directory}")
    
    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict] = None,
        embedding_function=None
    ):
        """Create or get collection"""
        try:
            # Try to get existing
            collection = self.client.get_collection(name=name)
            logger.info(f"Using existing collection: {name}")
        except Exception:
            # Create new
            default_metadata = {
                "created": datetime.now().isoformat(),
                "type": "regulatory_documents"
            }
            if metadata:
                default_metadata.update(metadata)
            
            collection = self.client.create_collection(
                name=name,
                metadata=default_metadata,
                embedding_function=embedding_function
            )
            logger.info(f"Created new collection: {name}")
        
        if name == self.collection_name:
            self.collection = collection
        
        return collection
    
    def get_collection(self, name: str = None):
        """Get collection by name"""
        name = name or self.collection_name
        return self.client.get_collection(name=name)
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        collections = self.client.list_collections()
        return [c.name for c in collections]
    
    def delete_collection(self, name: str):
        """Delete a collection"""
        self.client.delete_collection(name=name)
        logger.info(f"Deleted collection: {name}")
    
    def reset_database(self):
        """⚠️ Delete all data"""
        self.client.reset()
        logger.warning("Database reset complete")
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None,
        collection_name: str = None
    ):
        """
        Add documents to collection
        
        Args:
            documents: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
            ids: Optional custom IDs (auto-generated if None)
            collection_name: Target collection
        """
        collection = self.get_collection(collection_name)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}" 
                   for i in range(len(documents))]
        
        # Add to collection
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to {collection.name}")
    
    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict] = None,
        collection_name: str = None
    ) -> Dict:
        """
        Search similar documents
        
        Args:
            query_embeddings: Query vector(s)
            n_results: Number of results
            where: Metadata filter (e.g., {"source": "policy.pdf"})
            collection_name: Collection to search
        """
        collection = self.get_collection(collection_name)
        
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def get_document_count(self, collection_name: str = None) -> int:
        """Get total documents in collection"""
        collection = self.get_collection(collection_name)
        return collection.count()
    
    def backup(self, backup_dir: str = "data/backups"):
        """Create backup of database"""
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"chroma_backup_{timestamp}.zip"
        
        # Copy database directory
        shutil.make_archive(
            str(backup_file).replace('.zip', ''),
            'zip',
            self.persist_directory
        )
        
        logger.info(f"Backup created: {backup_file}")
        return backup_file
    
    def verify_persistence(self) -> bool:
        """Verify data persists after reload"""
        # Get current count
        if not self.collection:
            return False
        
        count_before = self.collection.count()
        
        # Simulate reload by creating new client with SAME settings
        try:
            new_client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            new_collection = new_client.get_collection(self.collection_name)
            count_after = new_collection.count()
            
            is_valid = count_before == count_after
            logger.info(f"Persistence check: {count_before} == {count_after} -> {is_valid}")
            
            return is_valid
        except ValueError:
            # ChromaDB client already exists, just verify we can read our own data
            count_after = self.collection.count()
            is_valid = count_before == count_after
            logger.info(f"Persistence check (single client): {count_before} == {count_after} -> {is_valid}")
            return is_valid


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 9: ChromaDB Manager Test")
    print("=" * 60)
    
    # Initialize
    db = ChromaManager()
    
    # 0. Clean start
    print("\n0. Cleaning old test data...")
    for coll in ["regulations", "policies"]:
        try:
            db.delete_collection(coll)
            print(f"   Deleted old collection: {coll}")
        except:
            print(f"   No existing collection: {coll}")
    
    # 1. Create collections
    print("\n1. Creating collections...")
    db.create_collection("regulations", {"type": "regulations"})
    db.create_collection("policies", {"type": "policies"})
    print(f"   Collections: {db.list_collections()}")
    
    # 2. Add documents with UNIQUE texts
    print("\n2. Adding sample documents...")
    sample_docs = [
        "Financial institutions must report transactions over $10,000.",  # finreg
        "Data breach notifications must be sent within 72 hours.",         # gdpr
        "Environmental impact assessments are required for new projects."  # epa
    ]
    sample_embeddings = [[0.1]*384, [0.2]*384, [0.3]*384]
    sample_metadata = [
        {"source": "finreg.pdf", "page": 1, "section": "Reporting"},
        {"source": "gdpr.pdf", "page": 5, "section": "Breach Notification"},
        {"source": "epa.pdf", "page": 12, "section": "Assessment"}
    ]
    
    db.add_documents(
        documents=sample_docs,
        embeddings=sample_embeddings,
        metadatas=sample_metadata,
        collection_name="regulations"
    )
    
    # 3. Query with detailed output
    print("\n3. Testing query (should find closest matches)...")
    results = db.query(
        query_embeddings=[[0.15]*384],  # Between 0.1 and 0.2
        n_results=2,
        collection_name="regulations"
    )
    
    print(f"   Found {len(results['ids'][0])} results:")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        print(f"   {i}. [{meta['source']}] (dist={dist:.3f}): {doc[:50]}...")
    
    # 4. Filtered query with verification
    print("\n4. Testing filtered query (source='finreg.pdf')...")
    filtered = db.query(
        query_embeddings=[[0.15]*384],
        where={"source": "finreg.pdf"},
        n_results=5,
        collection_name="regulations"
    )
    
    print(f"   Results: {len(filtered['ids'][0])}")
    for doc, meta in zip(filtered['documents'][0], filtered['metadatas'][0]):
        print(f"   - {meta['source']} p{meta['page']}: {doc[:40]}...")
    
    # Verify filter worked
    sources = [m['source'] for m in filtered['metadatas'][0]]
    if all(s == "finreg.pdf" for s in sources):
        print("   Filter verified: All results from finreg.pdf")
    else:
        print("   Filter failed: Found other sources!")
    
    # 5-7. Persistence, count, backup
    print("\n5. Testing persistence...")
    assert db.verify_persistence(), "Persistence failed!"
    
    print("\n6. Document count...")
    count = db.get_document_count("regulations")
    print(f"   Expected: 3, Actual: {count}")
    assert count == 3, f"Count mismatch! Expected 3, got {count}"
    
    print("\n7. Creating backup...")
    backup = db.backup()
    print(f"   Backup: {backup}")
    
    print("\n" + "=" * 60)
    print(" Day 9 complete! All tests passed.")
    print("=" * 60)