// Create the vector search index required by the RAG retriever.
//
// Option 1 — Atlas UI:
//   Database Deployments → your cluster → Search → Create Search Index
//   → JSON Editor → paste the definition below → Save
//
// Option 2 — mongosh (Atlas cluster or local MongoDB 7+):
//   mongosh "<YOUR_MONGODB_URI>" --eval "$(cat infra/mongo_index.js)"
//
// Option 3 — Atlas CLI:
//   atlas clusters search indexes create --clusterName <name> --file infra/mongo_index.js
//
// The index must be in READY state before ingest or query calls will return results.

db.documents.createSearchIndex({
  name: "vector_index",
  type: "vectorSearch",
  definition: {
    fields: [
      {
        type: "vector",
        path: "embedding",
        numDimensions: 768,
        similarity: "cosine",
      },
    ],
  },
});
