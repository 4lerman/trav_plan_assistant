import os
import sys
import argparse
import uuid

# Make sure we can import from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.corpus.schema import DestinationDoc


def get_docs(corpus_path=None):
    if corpus_path:
        import json

        with open(corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [DestinationDoc.model_validate(d) for d in data]
    else:
        from rag.corpus.sample_destinations import SAMPLE_DOCS

        return SAMPLE_DOCS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", help="path to external json file", default=None)
    args = parser.parse_args()

    docs = get_docs(args.corpus)
    if not docs:
        print("No documents to ingest.")
        return

    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    from FlagEmbedding import BGEM3FlagModel

    # Initialize Qdrant Client
    client = QdrantClient(url="http://localhost:6333", check_compatibility=False)

    collection_name = "destinations"

    # Create collection if not exists
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print(f"Created collection {collection_name}")

    # Initialize Embedding Model
    print("Loading BGEM3FlagModel...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    # Prepare points
    print(f"Embedding {len(docs)} documents...")
    sentences = [doc.description for doc in docs]
    output = model.encode(sentences, return_dense=True, return_sparse=False)
    dense_vecs = output["dense_vecs"]

    points = []
    for i, doc in enumerate(docs):
        # Qdrant accepts string UUIDs or integers
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.doc_id))
        points.append(
            PointStruct(
                id=point_id, vector=dense_vecs[i].tolist(), payload=doc.model_dump()
            )
        )

    print(f"Upserting {len(points)} points into Qdrant...")
    client.upsert(collection_name=collection_name, points=points)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
