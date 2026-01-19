import re
import pandas as pd # type: ignore
from pathlib import Path
from sentence_transformers import SentenceTransformer # type: ignore
from chromadb import PersistentClient # type: ignore
from chromadb.utils import embedding_functions # type: ignore
import numpy as np # type: ignore
from sklearn.cluster import KMeans, MiniBatchKMeans # type: ignore
import os
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import umap.umap_ as umap # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from plots import embeddings_map2d

MD_DIR = Path("marker-parsed/markdown")
CHROMA_PATH = "chroma_db"
N_CLUSTERS = 20
# Configure maximum chunk size for chroma.add via env var or default to 4000
CHROMA_MAX_BATCH = int(os.getenv("CHROMA_MAX_BATCH", "4000"))  # safe batch size for Chromadb adds

def chunked_add(collection, embeddings, documents, metadatas, ids, max_batch=CHROMA_MAX_BATCH):
    """Add items to a chroma collection in chunks to avoid hitting max batch size limits.

    embeddings: numpy array (n, dim) or list of lists
    documents: list[str]
    metadatas: list[dict]
    ids: list[str]
    """
    n = len(ids)
    print(f"Adding {n} items to collection '{collection.name}' in chunks of up to {max_batch}...")
    for i in range(0, n, max_batch):
        j = min(i + max_batch, n)
        emb_chunk = embeddings[i:j]
        # convert numpy arrays to native lists for Chroma
        try:
            emb_chunk_list = emb_chunk.tolist()
        except Exception:
            emb_chunk_list = list(emb_chunk)

        print(f"  - adding items {i}..{j} ({j-i} items)")
        collection.add(
            embeddings=emb_chunk_list,
            documents=documents[i:j],
            metadatas=metadatas[i:j],
            ids=ids[i:j],
        )

def chunked_add(collection, embeddings, documents, metadatas, ids, max_batch=CHROMA_MAX_BATCH):
    """Add items to a chroma collection in chunks to avoid hitting max batch size limits.

    embeddings: numpy array (n, dim) or list of lists
    documents: list[str]
    metadatas: list[dict]
    ids: list[str]
    """
    n = len(ids)
    # ensure embeddings is a list of lists for chroma
    # If embeddings is numpy array, slice and call .tolist() per chunk to avoid large memory spikes
    for i in range(0, n, max_batch):
        j = min(i + max_batch, n)
        emb_chunk = embeddings[i:j]
        # if numpy array, convert to list
        try:
            emb_chunk_list = emb_chunk.tolist()
        except Exception:
            emb_chunk_list = list(emb_chunk)

        collection.add(
            embeddings=emb_chunk_list,
            documents=documents[i:j],
            metadatas=metadatas[i:j],
            ids=ids[i:j],
        )

# parse markdown files into hierarchical sections
def parse_md_sections(md_path: Path):
    content = md_path.read_text()
    sections = []
    current_section = {'level': 0, 'title': 'default title', 'content': ''}

    # split the document into lines and identify titles and content
    for line in content.split('\n'):
        title_match = re.match(r'^(#{1,6})\s+(.+)', line) # up to 6 levels
        if title_match:
            if current_section['content'].strip():
                sections.append(current_section)
            level = len(title_match.group(1))
            current_section = {
                'level': level,
                'title': title_match.group(2),
                'content': ''
            }
        else:
            current_section['content'] += line + '\n'
    if current_section['content'].strip():
        sections.append(current_section)
    return sections

def title_embeddings():
    rows = []
    for md_path in sorted(MD_DIR.glob("*.md")):
        doc_id = md_path.stem  # e.g. "491_1"
        sections = parse_md_sections(md_path)
        for i, s in enumerate(sections):
            sec_id = f"{doc_id}_sec_{i}"
            rows.append(
                {
                    "doc_id": doc_id,
                    "sec_id": sec_id,
                    "level": s["level"],
                    "title": s["title"],
                    "content": s["content"],
                    "text": f"{s['title']} {s['content']}",
                }
            )

    df = pd.DataFrame(rows)

    # prepare data for streaming encode + add
    model = SentenceTransformer('all-MiniLM-L6-v2')
    BATCH_SIZE = 256

    texts = df["title"].tolist()
    metadatas = [
        {
            "doc_id": r["doc_id"],
            "sec_id": r["sec_id"],
            "level": r["level"],
            "title": r["title"],
        }
        for r in df.to_dict(orient="records")
    ]
    ids = df["sec_id"].tolist()

    # store in ChromaDB with metadata (create collection once)
    chroma_client = PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
    collection = chroma_client.get_or_create_collection(
        name="md_sections",
        embedding_function=ef
    )

    # incremental clustering with MiniBatchKMeans to avoid storing all embeddings
    n_samples = len(texts)
    clusters_n = min(N_CLUSTERS, max(1, n_samples))
    mbk = MiniBatchKMeans(n_clusters=clusters_n, random_state=16, batch_size=BATCH_SIZE)
    all_clusters = []
    all_embeddings = []

    # encode in batches and add in chunks as we go
    for i in range(0, len(texts), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(texts))
        batch_texts = texts[i:j]
        print(f"Encoding titles {i}:{j} / {len(texts)}")
        emb = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.extend(emb.tolist())

        # update clustering model incrementally and predict cluster for this batch
        try:
            mbk.partial_fit(emb)
            batch_clusters = mbk.predict(emb)
        except Exception:
            # fallback: if partial_fit not applicable (very small data), assign zeros
            batch_clusters = [0] * len(batch_texts)

        all_clusters.extend(batch_clusters)

        # add this batch to chroma (chunked_add will further split if needed)
        chunked_add(collection, emb, batch_texts, metadatas[i:j], ids[i:j])

    df['cluster'] = all_clusters
    df['embedding'] = all_embeddings

    # save structured data
    os.makedirs("title_embeddings", exist_ok=True)
    df.to_parquet('title_embeddings/all_md_section_titles.parquet')

def section_embeddings(): # a similar function for paragraph embeddings
    rows = []
    for md_path in sorted(MD_DIR.glob("*.md")):
        doc_id = md_path.stem  # e.g. "491_1"
        sections = parse_md_sections(md_path)
        for i, s in enumerate(sections):
            sec_id = f"{doc_id}_sec_{i}"
            rows.append(
                {
                    "doc_id": doc_id,
                    "sec_id": sec_id,
                    "level": s["level"],
                    "title": s["title"],
                    "content": s["content"],
                    "text": f"{s['title']} {s['content']}",
                }
            )

    df = pd.DataFrame(rows)

    # generate embeddings in streaming fashion and add to chroma incrementally
    model = SentenceTransformer('all-MiniLM-L6-v2')
    BATCH_SIZE = 256

    texts = df["text"].tolist()
    metadatas = [
        {
            "doc_id": r["doc_id"],
            "sec_id": r["sec_id"],
            "level": r["level"],
            "title": r["title"],
        }
        for r in df.to_dict(orient="records")
    ]
    ids = df["sec_id"].tolist()

    chroma_client = PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
    collection = chroma_client.get_or_create_collection(
        name="md_sections",
        embedding_function=ef
    )

    # incremental clustering
    n_samples = len(texts)
    clusters_n = min(N_CLUSTERS, max(1, n_samples))
    mbk = MiniBatchKMeans(n_clusters=clusters_n, random_state=16, batch_size=BATCH_SIZE)
    all_clusters = []
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(texts))
        batch_texts = texts[i:j]
        print(f"Encoding sections {i}:{j} / {len(texts)}")
        emb = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.extend(emb.tolist())

        try:
            mbk.partial_fit(emb)
            batch_clusters = mbk.predict(emb)
        except Exception:
            batch_clusters = [0] * len(batch_texts)

        all_clusters.extend(batch_clusters)

        chunked_add(collection, emb, batch_texts, metadatas[i:j], ids[i:j])

    df['cluster'] = all_clusters
    df['embedding'] = all_embeddings

    # save structured data
    os.makedirs("section_embeddings", exist_ok=True)
    df.to_parquet('section_embeddings/all_md_sections.parquet')

def search_markdown_folder(folder, keyword, case_sensitive=False): # update to load all texts once and then search them all 
    folder = Path(folder)
    if not case_sensitive:
        keyword = keyword.lower()

    total_matches = 0
    files_with_matches = 0
    total_files = 0
    file_ids = []

    for path in folder.rglob("*.md"):  # recursively search for .md files
        total_files += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Could not read {path}: {e}")
            continue

        haystack = text if case_sensitive else text.lower()
        count = haystack.count(keyword)
        if count > 0:
            files_with_matches += 1
            total_matches += count
            file_ids.append(path.stem)  # store the file ID (stem) for matched files

    return {
        "keyword": keyword,
        "total_files": total_files,
        "files_with_matches": files_with_matches,
        "total_matches": total_matches
    }

BOOL_OPS = {"AND", "OR", "NOT"}

def file_term_presence(text, terms, case_sensitive=False):
    if not case_sensitive:
        text = text.lower() # text is the entire content of the file
    presence = {}
    for t in terms:
        key = t if case_sensitive else t.lower()
        presence[t] = (key in text)
    return presence

def parse_boolean_query(raw_query):
    """
    Convert a user string like:
        "apple AND (banana OR NOT cherry)"
    into a Python expression like:
        "vars['apple'] and (vars['banana'] or (not vars['cherry']))"
    and return (expr_string, set_of_terms).
    """
    # Normalize spaces
    q = raw_query.strip()

    # Tokenize: words and parentheses
    tokens = re.findall(r'\w+|\(|\)', q)

    terms = set()
    py_tokens = []

    for token in tokens:
        upper = token.upper()
        if upper in BOOL_OPS:
            # map to Python boolean operators
            if upper == "AND":
                py_tokens.append("and")
            elif upper == "OR":
                py_tokens.append("or")
            elif upper == "NOT":
                py_tokens.append("not")
        elif token in ("(", ")"):
            py_tokens.append(token)
        else:
            # a search term
            terms.add(token)
            py_tokens.append(f"vars[{token!r}]")  # e.g. vars['apple']

    expr = " ".join(py_tokens)
    return expr, terms

def boolean_search_markdown(folder, query, case_sensitive=False):
    folder = Path(folder)

    # build Python expression and term set from the query
    expr, terms = parse_boolean_query(query)

    total_files = 0
    files_matching_query = 0

    for path in folder.rglob("*.md"):
        total_files += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Could not read {path}: {e}")
            continue

        # For each file, compute which terms are present
        vars_for_file = file_term_presence(text, terms, case_sensitive=case_sensitive)

        # Evaluate the boolean expression in a restricted environment
        try:
            match = bool(eval(expr, {"__builtins__": {}}, {"vars": vars_for_file}))
        except Exception as e:
            print(f"Error evaluating query on {path}: {e}")
            continue

        if match:
            files_matching_query += 1

    return {
        "query": query,
        "total_files": total_files,
        "files_matching_query": files_matching_query,
    }


def search_markdown_folder_freq(folder, keyword, case_sensitive=False, max_words=3): # includes per file frequencies
    folder = Path(folder)
    if not case_sensitive:
        keyword = keyword.lower()
    
    # pre-tokenise keyword to know its length
    #keyword_words = keyword.split()
    #keyword_len = len(keyword_words)
    
    total_matches = 0
    files_with_matches = 0
    total_files = 0
    file_results = [] 
    
    for path in folder.rglob("*.md"):
        total_files += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Could not read {path}: {e}")
            continue
        
        haystack = text if case_sensitive else text.lower()
        words = re.findall(r'\b\w+\b', haystack)
        
        #if keyword_len == 1:
            # single word: use fast count
            #count = haystack.count(keyword)
        #else:
            # multi-word phrase: sliding window
            #count = 0
            #for i in range(len(words) - keyword_len + 1):
                #ngram = ' '.join(words[i:i+keyword_len])
                #if ngram == keyword:
                    #count += 1
        count = haystack.count(keyword)

        if count > 0:
            files_with_matches += 1
            total_matches += count
            file_results.append({
                "file_id": path.stem,
                "frequency": count
            })
    
    return {
        "keyword": keyword,
        "total_files": total_files,
        "files_with_matches": files_with_matches,
        "total_matches": total_matches,
        "file_frequencies": file_results  # New: list of dicts with per-file counts
    }

def visualise_file_frequencies(result):
    file_data = result['file_frequencies']
    if not file_data:
        print("No matches found to visualise.")
        return
    
    # sort by frequency (highest first)
    file_data_sorted = sorted(file_data, key=lambda x: x['frequency'], reverse=True)
    
    files = [f['file_id'] for f in file_data_sorted]
    frequencies = [f['frequency'] for f in file_data if 1 < f['frequency'] < 50]

    freq_counter = {}
    for freq in frequencies:
        freq_counter[freq] = freq_counter.get(freq, 0) + 1
    
    # sort by frequency value
    sorted_freqs = sorted(freq_counter.items())
    freq_values = [f[0] for f in sorted_freqs]
    file_counts = [f[1] for f in sorted_freqs]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(freq_values, file_counts, color='skyblue', edgecolor='navy', alpha=0.8)
    
    plt.title(f'"{result["keyword"]}" Frequency Distribution Across Files\n'
              f'(Total: {result["total_matches"]} matches in {result["files_with_matches"]} files)')
    plt.xlabel('Keyword Occurrences per File')
    plt.ylabel('Number of Files')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, file_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save with safe filename
    safe_keyword = result["keyword"].replace(" ", "_").replace("/", "_")
    os.makedirs("keyword_frequency_histograms", exist_ok=True)
    save_path = f'keyword_frequency_histograms/keyword_frequency_histogram_{safe_keyword}_v2.png' #v2 has the 2-word counter commented out
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # Load embeddings from parquet
    df = pd.read_parquet("title_embeddings/all_md_section_titles.parquet")
    # if embeddings are stored as lists/arrays in a column
    X = np.vstack(df["embedding"].to_numpy())  # shape: (n_samples, dim)
    embeddings_map2d(n_clusters=N_CLUSTERS, random_state=16, top_k=10, X=X)