"""
Data Management Module
Handles loading and saving of embeddings and metadata
"""

import os
import numpy as np
import pickle
from typing import List, Tuple, Optional
from urllib.parse import urlparse

# Try to import torch, fallback to numpy if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Paths
DATA_DIR = "data"
SCRAPED_METADATA_FILE = os.path.join(DATA_DIR, "embeddings_metadata.pkl")
SCRAPED_EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
STRUCTURED_METADATA_FILE = os.path.join(DATA_DIR, "structured_metadata.pkl")
STRUCTURED_EMBEDDINGS_FILE = os.path.join(DATA_DIR, "structured_embeddings.npy")


def save_embeddings_data(pages_data, embeddings, metadata_file=None, embeddings_file=None):
    """Save scraped data as embeddings and metadata"""
    if metadata_file is None:
        metadata_file = SCRAPED_METADATA_FILE
    if embeddings_file is None:
        embeddings_file = SCRAPED_EMBEDDINGS_FILE
    
    # Ensure directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    try:
        # Save metadata (URLs and texts) as pickle
        with open(metadata_file, "wb") as f:
            pickle.dump(pages_data, f)
        print(f"✅ Saved metadata to {metadata_file} ({len(pages_data)} pages)")
        
        # Save embeddings as numpy array
        if embeddings is not None:
            # Convert to numpy if it's a torch tensor
            if TORCH_AVAILABLE and hasattr(embeddings, 'cpu'):
                embeddings_np = embeddings.cpu().numpy()
            elif hasattr(embeddings, 'numpy'):
                embeddings_np = embeddings.numpy()
            else:
                embeddings_np = np.array(embeddings)
            
            np.save(embeddings_file, embeddings_np)
            print(f"✅ Saved embeddings to {embeddings_file} (shape: {embeddings_np.shape})")
        
        return True
    except Exception as e:
        print(f"❌ Error saving embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_embeddings_data(metadata_file=None, embeddings_file=None, embedder=None):
    """Load scraped data from embeddings and metadata files"""
    if metadata_file is None:
        metadata_file = SCRAPED_METADATA_FILE
    if embeddings_file is None:
        embeddings_file = SCRAPED_EMBEDDINGS_FILE
    
    try:
        # Load metadata
        if not os.path.exists(metadata_file):
            print(f"⚠️ Metadata file {metadata_file} not found")
            return [], None
        
        with open(metadata_file, "rb") as f:
            pages_data = pickle.load(f)
        
        if not pages_data:
            print(f"⚠️ No data in {metadata_file}")
            return [], None
        
        # Load embeddings
        embeddings_np = None
        embeddings = None
        if os.path.exists(embeddings_file):
            embeddings_np = np.load(embeddings_file)
            print(f"Loaded embeddings from {embeddings_file} (shape: {embeddings_np.shape})")
            
            # Convert to tensor if torch is available
            if TORCH_AVAILABLE:
                embeddings = torch.from_numpy(embeddings_np)
            else:
                embeddings = embeddings_np
        else:
            print(f"⚠️ Embeddings file {embeddings_file} not found, will recreate embeddings")
            # Recreate embeddings if embedder is available
            if embedder:
                texts = [text for _, text in pages_data]
                embeddings = embedder.encode(texts, convert_to_tensor=True)
                # Save the recreated embeddings
                save_embeddings_data(pages_data, embeddings, metadata_file, embeddings_file)
        
        print(f"✅ Loaded {len(pages_data)} pages with embeddings")
        return pages_data, embeddings
        
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        import traceback
        traceback.print_exc()
        return [], None


def save_structured_embeddings_data(texts, sources, embeddings, metadata_file=None, embeddings_file=None):
    """Save structured data as embeddings"""
    if metadata_file is None:
        metadata_file = STRUCTURED_METADATA_FILE
    if embeddings_file is None:
        embeddings_file = STRUCTURED_EMBEDDINGS_FILE
    
    # Ensure directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    try:
        # Save metadata
        with open(metadata_file, "wb") as f:
            pickle.dump({'texts': texts, 'sources': sources}, f)
        print(f"✅ Saved structured metadata to {metadata_file}")
        
        # Save embeddings
        if embeddings is not None:
            if TORCH_AVAILABLE and hasattr(embeddings, 'cpu'):
                embeddings_np = embeddings.cpu().numpy()
            elif hasattr(embeddings, 'numpy'):
                embeddings_np = embeddings.numpy()
            else:
                embeddings_np = np.array(embeddings)
            
            np.save(embeddings_file, embeddings_np)
            print(f"✅ Saved structured embeddings to {embeddings_file} (shape: {embeddings_np.shape})")
        
        return True
    except Exception as e:
        print(f"❌ Error saving structured embeddings: {e}")
        return False


def load_structured_embeddings_data(metadata_file=None, embeddings_file=None, embedder=None):
    """Load structured data from embeddings"""
    if metadata_file is None:
        metadata_file = STRUCTURED_METADATA_FILE
    if embeddings_file is None:
        embeddings_file = STRUCTURED_EMBEDDINGS_FILE
    
    try:
        if not os.path.exists(metadata_file):
            return [], [], None
        
        with open(metadata_file, "rb") as f:
            data = pickle.load(f)
        
        structured_data_texts = data.get('texts', [])
        structured_data_sources = data.get('sources', [])
        
        # Load embeddings
        embeddings = None
        if os.path.exists(embeddings_file):
            embeddings_np = np.load(embeddings_file)
            if TORCH_AVAILABLE:
                embeddings = torch.from_numpy(embeddings_np)
            else:
                embeddings = embeddings_np
            print(f"✅ Loaded structured embeddings (shape: {embeddings_np.shape})")
        else:
            # Recreate embeddings
            if embedder and structured_data_texts:
                embeddings = embedder.encode(structured_data_texts, convert_to_tensor=True)
                save_structured_embeddings_data(structured_data_texts, structured_data_sources, embeddings)
        
        return structured_data_texts, structured_data_sources, embeddings
    except Exception as e:
        print(f"❌ Error loading structured embeddings: {e}")
        return [], [], None


def embeddings_match_website(pages_data, target_website):
    """
    Check if the loaded embeddings match the target website.
    Returns True if embeddings match, False otherwise.
    """
    if not pages_data or not target_website:
        return False
    
    try:
        # Parse target website domain
        target_parsed = urlparse(target_website)
        target_domain = target_parsed.netloc.lower()
        if target_domain.startswith('www.'):
            target_domain = target_domain[4:]
        
        # Check if any URL in pages_data matches the target domain
        for url, _ in pages_data[:5]:  # Check first 5 URLs
            try:
                url_parsed = urlparse(url)
                url_domain = url_parsed.netloc.lower()
                if url_domain.startswith('www.'):
                    url_domain = url_domain[4:]
                
                # Check if domains match
                if url_domain == target_domain or url_domain.endswith('.' + target_domain):
                    return True
            except Exception:
                continue
        
        return False
    except Exception as e:
        print(f"⚠️ Error checking website match: {e}")
        return False

