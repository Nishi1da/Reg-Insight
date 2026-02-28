"""Persistence Manager - Backup, restore, and version control"""

import json
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersistenceManager:
    def __init__(
        self,
        chroma_path: str = "data/processed/chroma_db",
        backup_dir: str = "data/backups",
        version_file: str = "data/versions.json"
    ):
        self.chroma_path = Path(chroma_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.version_file = Path(version_file)
        
        # Initialize version tracking
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict:
        """Load version history"""
        if self.version_file.exists():
            with open(self.version_file) as f:
                return json.load(f)
        return {"versions": [], "current": None}
    
    def _save_versions(self):
        """Save version history"""
        self.version_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def _generate_version_id(self) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]
        return f"v_{timestamp}_{random_suffix}"
    
    def _convert_to_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to Python types"""
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return float(obj) if isinstance(obj, (np.float64, np.float32)) else int(obj)
        return obj
    
    def backup(
        self,
        name: Optional[str] = None,
        include_embeddings: bool = True
    ) -> Path:
        """
        Create full backup of vector store
        
        Args:
            name: Optional backup name
            include_embeddings: Include full embedding vectors (large)
        """
        version_id = self._generate_version_id()
        timestamp = datetime.now().isoformat()
        
        backup_name = name or f"backup_{version_id}"
        backup_path = self.backup_dir / f"{backup_name}.zip"
        
        # Create manifest
        manifest = {
            "version_id": version_id,
            "created": timestamp,
            "name": name or "unnamed",
            "include_embeddings": include_embeddings,
            "chroma_path": str(self.chroma_path)
        }
        
        # Create temporary directory for backup contents
        temp_dir = self.backup_dir / f"temp_{version_id}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Write manifest
            with open(temp_dir / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Copy ChromaDB (or export without embeddings if specified)
            if include_embeddings:
                shutil.copytree(self.chroma_path, temp_dir / "chroma_db")
            else:
                # Export metadata only
                self._export_metadata_only(temp_dir / "metadata_export.json")
            
            # Create zip
            shutil.make_archive(
                str(backup_path).replace('.zip', ''),
                'zip',
                temp_dir
            )
            
            # Update version history
            self.versions["versions"].append({
                "id": version_id,
                "name": name,
                "created": timestamp,
                "path": str(backup_path),
                "size_mb": backup_path.stat().st_size / (1024 * 1024)
            })
            self.versions["current"] = version_id
            self._save_versions()
            
            manifest['size_mb'] = backup_path.stat().st_size / (1024 * 1024)
            logger.info(f"Backup created: {backup_path} ({manifest['size_mb']:.1f} MB)")
            return backup_path
            
        finally:
            # Cleanup temp
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _export_metadata_only(self, output_path: Path):
        """Export only metadata (no embeddings) for smaller backup"""
        import chromadb
        
        client = chromadb.PersistentClient(path=str(self.chroma_path))
        collections = client.list_collections()
        
        export_data = {"collections": []}
        
        for collection in collections:
            data = collection.get()
            coll_export = {
                "name": collection.name,
                "metadata": collection.metadata,
                "documents": data['documents'],
                "metadatas": data['metadatas'],
                "ids": data['ids']
                # Note: embeddings excluded
            }
            export_data["collections"].append(coll_export)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def restore(self, backup_path: Path, force: bool = False):
        """
        Restore from backup
        
        Args:
            backup_path: Path to backup zip file
            force: Overwrite existing data without confirmation
        """
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        if not force and self.chroma_path.exists():
            response = input(f"Overwrite existing data at {self.chroma_path}? (y/n): ")
            if response.lower() != 'y':
                logger.info("Restore cancelled")
                return False
        
        # Extract to temp
        temp_dir = Path("data/temp_restore")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(backup_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Read manifest
            manifest_path = temp_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                logger.info(f"Restoring: {manifest.get('name', 'unknown')}")
            
            # Restore ChromaDB
            chroma_backup = temp_dir / "chroma_db"
            if chroma_backup.exists():
                if self.chroma_path.exists():
                    shutil.rmtree(self.chroma_path)
                shutil.copytree(chroma_backup, self.chroma_path)
                logger.info(f"Restored to: {self.chroma_path}")
            else:
                # Restore from metadata export
                metadata_file = temp_dir / "metadata_export.json"
                if metadata_file.exists():
                    self._import_from_metadata(metadata_file)
            
            return True
            
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _import_from_metadata(self, metadata_path: Path):
        """Reconstruct collection from metadata (embeddings will be regenerated)"""
        import chromadb
        from embeddings.embedding_generator import EmbeddingGenerator
        
        with open(metadata_path) as f:
            data = json.load(f)
        
        client = chromadb.PersistentClient(path=str(self.chroma_path))
        embedder = EmbeddingGenerator()
        
        for coll_data in data["collections"]:
            collection = client.create_collection(
                name=coll_data["name"],
                metadata=coll_data["metadata"]
            )
            
            # Regenerate embeddings
            logger.info(f"Regenerating embeddings for {coll_data['name']}...")
            embeddings = embedder.encode(coll_data["documents"], show_progress=True)
            
            collection.add(
                documents=coll_data["documents"],
                embeddings=embeddings.tolist(),
                metadatas=coll_data["metadatas"],
                ids=coll_data["ids"]
            )
    
    def list_backups(self) -> List[Dict]:
        """List all available backups"""
        return self.versions.get("versions", [])
    
    def export_to_json(
        self,
        output_path: str = "exports/vector_store.json",
        collection_name: Optional[str] = None
    ):
        """Export collection to JSON format"""
        import chromadb
        
        client = chromadb.PersistentClient(path=str(self.chroma_path))
        
        if collection_name:
            collections = [client.get_collection(collection_name)]
        else:
            collections = client.list_collections()
        
        export_data = {"exported_at": datetime.now().isoformat(), "collections": []}
        
        for collection in collections:
            data = collection.get(include=["documents", "metadatas", "embeddings"])
            
            coll_export = {
                "name": collection.name,
                "metadata": collection.metadata,
                "count": len(data['ids']),
                "items": []
            }
            
            for i in range(len(data['ids'])):
                # FIX: Convert embedding to serializable format
                embedding = data['embeddings'][i] if 'embeddings' in data else None
                if embedding is not None:
                    embedding = self._convert_to_serializable(embedding)
                
                item = {
                    "id": data['ids'][i],
                    "document": data['documents'][i],
                    "metadata": data['metadatas'][i],
                    "embedding": embedding
                }
                coll_export["items"].append(item)
            
            export_data["collections"].append(coll_export)
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported to: {output_path}")
        return Path(output_path)
    
    def export_to_csv(
        self,
        output_path: str = "exports/documents.csv",
        collection_name: Optional[str] = None
    ):
        """Export to CSV (metadata only, no embeddings)"""
        import chromadb
        import csv
        
        client = chromadb.PersistentClient(path=str(self.chroma_path))
        collection = client.get_collection(collection_name or "regulations")
        data = collection.get()
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if data['metadatas']:
                fieldnames = ['id', 'document'] + list(data['metadatas'][0].keys())
            else:
                fieldnames = ['id', 'document']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for i in range(len(data['ids'])):
                row = {
                    'id': data['ids'][i],
                    'document': data['documents'][i]
                }
                if data['metadatas']:
                    row.update(data['metadatas'][i])
                writer.writerow(row)
        
        logger.info(f"Exported CSV to: {output_path}")
        return Path(output_path)
    
    def compact(self):
        """Remove deleted documents and optimize storage"""
        import chromadb
        
        client = chromadb.PersistentClient(path=str(self.chroma_path))
        collections = client.list_collections()
        
        total_before = sum(c.count() for c in collections)
        
        # ChromaDB doesn't have a direct compact, but we can recreate collections
        # to remove any fragmentation
        for collection in collections:
            data = collection.get()
            if len(data['ids']) == 0:
                client.delete_collection(collection.name)
                logger.info(f"Removed empty collection: {collection.name}")
        
        total_after = sum(c.count() for c in client.list_collections())
        logger.info(f"Compaction complete: {total_before} → {total_after} documents")
    
    def migrate_schema(self, old_version: str, new_version: str):
        """Handle schema migrations between versions"""
        # Example migration logic
        migrations = {
            ("1.0", "1.1"): self._migrate_1_0_to_1_1
        }
        
        key = f"{old_version}->{new_version}"
        if key in migrations:
            migrations[key]()
            logger.info(f"Migrated schema: {old_version} → {new_version}")
        else:
            logger.warning(f"No migration path: {old_version} → {new_version}")
    
    def _migrate_1_0_to_1_1(self):
        """Example: Add new metadata field to all documents"""
        pass


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 13: Persistence Manager Test")
    print("=" * 60)
    
    pm = PersistenceManager()
    
    # Test 1: Backup
    print("\n1. Creating backup...")
    backup_path = pm.backup(name="test_backup", include_embeddings=True)
    print(f"   Created: {backup_path}")
    
    # Test 2: List backups
    print("\n2. Listing backups...")
    backups = pm.list_backups()
    for b in backups:
        print(f"   - {b['name']} ({b['created']})")
    
    # Test 3: Export JSON
    print("\n3. Exporting to JSON...")
    json_path = pm.export_to_json("exports/test_export.json")
    print(f"   Exported: {json_path}")
    
    # Test 4: Export CSV
    print("\n4. Exporting to CSV...")
    csv_path = pm.export_to_csv("exports/test_export.csv")
    print(f"   Exported: {csv_path}")
    
    # Test 5: Compact
    print("\n5. Running compaction...")
    pm.compact()
    
    print("\n" + "=" * 60)
    print(" Day 13 complete!")
    print("=" * 60)