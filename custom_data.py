"""
Custom Data Management Module
Handles loading, saving, adding, and removing custom data entries
"""

import os
import numpy as np
import pickle
from typing import List, Tuple, Optional, Dict

# Try to import torch, fallback to numpy if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Global state for custom data
custom_data_texts: List[str] = []
custom_data_embeddings = None
custom_data_sources: List[Dict] = []

# Paths
CUSTOM_DATA_DIR = "custom_data"
CUSTOM_DATA_FILE = os.path.join(CUSTOM_DATA_DIR, "custom_data.txt")


def load_custom_data(embedder=None):
    """Load custom data from file if it exists - handles correct format"""
    global custom_data_texts, custom_data_embeddings, custom_data_sources
    
    # Ensure directory exists
    os.makedirs(CUSTOM_DATA_DIR, exist_ok=True)
    
    if not os.path.exists(CUSTOM_DATA_FILE):
        print(f"⚠️ Custom data file {CUSTOM_DATA_FILE} not found")
        return []
    
    try:
        with open(CUSTOM_DATA_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        
        if not content.strip():
            print(f"⚠️ Custom data file {CUSTOM_DATA_FILE} is empty")
            return []
        
        # Split by separator, but handle case where file starts with separator
        entries = content.split("--- CUSTOM ENTRY ---")
        custom_data_texts = []
        custom_data_sources = []
        
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            
            lines = entry.split('\n')
            if len(lines) >= 3:
                title = lines[0].strip()
                category = lines[1].strip()
                text = '\n'.join(lines[2:]).strip()
                
                if title and category and text:
                    custom_data_texts.append(text)
                    custom_data_sources.append({
                        'title': title,
                        'category': category,
                        'type': 'custom'
                    })
                else:
                    print(f"⚠️ Skipping invalid entry (missing title, category, or text)")
            else:
                print(f"⚠️ Skipping entry with insufficient lines (expected at least 3, got {len(lines)})")
        
        print(f"✅ Loaded {len(custom_data_texts)} custom data entries from file")
        
        # Create embeddings for custom data if embedder is available
        if embedder and custom_data_texts:
            try:
                print(f"🔄 Creating embeddings for {len(custom_data_texts)} custom data entries...")
                custom_data_embeddings = embedder.encode(custom_data_texts, convert_to_tensor=True)
                print(f"✅ Created embeddings for {len(custom_data_texts)} custom data entries")
            except Exception as e:
                print(f"❌ Error creating custom data embeddings: {e}")
                import traceback
                traceback.print_exc()
                custom_data_embeddings = None
        
        return custom_data_texts
    except Exception as e:
        print(f"❌ Error loading custom data: {e}")
        import traceback
        traceback.print_exc()
        return []


def save_custom_data():
    """Save custom data to file in correct format"""
    global custom_data_texts, custom_data_sources
    
    try:
        # Ensure directory exists
        os.makedirs(CUSTOM_DATA_DIR, exist_ok=True)
        
        if not custom_data_texts or len(custom_data_texts) == 0:
            # If no data, create empty file or remove existing
            if os.path.exists(CUSTOM_DATA_FILE):
                with open(CUSTOM_DATA_FILE, "w", encoding="utf-8") as f:
                    f.write("")
            return True
        
        with open(CUSTOM_DATA_FILE, "w", encoding="utf-8") as f:
            for i, (text, source) in enumerate(zip(custom_data_texts, custom_data_sources)):
                # Write entry: title, category, content
                f.write(f"{source['title']}\n")
                f.write(f"{source['category']}\n")
                f.write(f"{text}\n")
                # Only add separator between entries, not after the last one
                if i < len(custom_data_texts) - 1:
                    f.write("--- CUSTOM ENTRY ---\n")
        
        print(f"✅ Custom data saved to {CUSTOM_DATA_FILE} ({len(custom_data_texts)} entries)")
        return True
    except Exception as e:
        print(f"❌ Error saving custom data: {e}")
        import traceback
        traceback.print_exc()
        return False


def add_custom_data(title: str, category: str, content: str, embedder=None):
    """Add new custom data entry"""
    global custom_data_texts, custom_data_embeddings, custom_data_sources
    
    try:
        # Validate inputs
        title = title.strip() if title else ""
        category = category.strip() if category else ""
        content = content.strip() if content else ""
        
        if not title or not category or not content:
            return False, "All fields (title, category, content) are required and cannot be empty"
        
        # Add the new entry to lists first
        custom_data_texts.append(content)
        custom_data_sources.append({
            'title': title,
            'category': category,
            'type': 'custom'
        })
        
        # Update embeddings if embedder is available
        if embedder:
            try:
                # Create new embedding for the added content
                print(f"🔄 Creating embedding for custom data: {title}")
                new_embedding = embedder.encode([content], convert_to_tensor=True)
                
                if custom_data_embeddings is None:
                    custom_data_embeddings = new_embedding
                else:
                    # Concatenate with existing embeddings
                    if TORCH_AVAILABLE:
                        custom_data_embeddings = torch.cat([custom_data_embeddings, new_embedding], dim=0)
                    else:
                        # Fallback to numpy concatenation
                        custom_data_embeddings = np.concatenate([custom_data_embeddings, new_embedding], axis=0)
                
                print(f"✅ Added custom data with embedding: {title} ({category})")
                return True, "Custom data added successfully"
            except Exception as e:
                print(f"❌ Error creating embedding for new custom data: {e}")
                import traceback
                traceback.print_exc()
                # Remove the added entry if embedding fails
                if len(custom_data_texts) > 0:
                    custom_data_texts.pop()
                if len(custom_data_sources) > 0:
                    custom_data_sources.pop()
                return False, f"Error creating embedding: {str(e)}"
        else:
            print(f"✅ Added custom data (no embedding): {title} ({category})")
            return True, "Custom data added successfully (embeddings not available)"
    except Exception as e:
        print(f"❌ Error in add_custom_data: {e}")
        import traceback
        traceback.print_exc()
        # Clean up if something went wrong
        if len(custom_data_texts) > len(custom_data_sources):
            custom_data_texts.pop()
        elif len(custom_data_sources) > len(custom_data_texts):
            custom_data_sources.pop()
        return False, f"Error adding custom data: {str(e)}"


def remove_custom_data(index: int, embedder=None):
    """Remove custom data entry by index"""
    global custom_data_texts, custom_data_embeddings, custom_data_sources
    
    if index < 0 or index >= len(custom_data_texts):
        return False, "Invalid index"
    
    try:
        # Remove the entry
        removed_title = custom_data_sources[index]['title']
        custom_data_texts.pop(index)
        custom_data_sources.pop(index)
        
        # Recreate embeddings if embedder is available
        if embedder and custom_data_texts:
            try:
                custom_data_embeddings = embedder.encode(custom_data_texts, convert_to_tensor=True)
                print(f"✅ Recreated embeddings after removing custom data")
            except Exception as e:
                print(f"Error recreating embeddings: {e}")
                custom_data_embeddings = np.array([])
        elif not custom_data_texts:
            custom_data_embeddings = None
        
        print(f"Removed custom data: {removed_title}")
        return True, f"Removed: {removed_title}"
    except Exception as e:
        print(f"Error removing custom data: {e}")
        return False, f"Error removing data: {str(e)}"


def get_custom_data():
    """Get all custom data entries"""
    return custom_data_texts, custom_data_sources


def get_custom_data_embeddings():
    """Get custom data embeddings"""
    return custom_data_embeddings

