# Hybrid Face Recognition Examples

This project now supports embedding + geometry fusion through `/predict/`.

## 1) Example input images

Use any face images from your dataset structure. Example file paths:

- `data/vggface2/person_001/img_0001.jpg`
- `data/vggface2/person_002/img_0003.jpg`
- `data/vggface2/person_003/img_0010.jpg`

You can call the API with curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict/?top_k=3&enable_hybrid=true" \
  -F "file=@data/vggface2/person_001/img_0001.jpg"
```

## 2) Expected output JSON (name + scores)

Example response shape:

```json
{
  "predictions": [
    {
      "name": "person_001",
      "score": 0.9341,
      "class_prob": 0.9188,
      "class_index": 0,
      "embedding_similarity": 0.9042,
      "geometry_similarity": 0.8129
    },
    {
      "name": "person_017",
      "score": 0.2114,
      "class_prob": 0.0703,
      "class_index": 16,
      "embedding_similarity": 0.1037,
      "geometry_similarity": 0.0581
    }
  ],
  "embedding_dim": 512,
  "hybrid_enabled": true
}
```

## 3) Example accuracy results table

| Model variant     | Rank-1 Accuracy | Notes |
|------------------|-----------------|-------|
| Embedding only   | 91.3%           | Baseline using embedding branch/prototype matching only |
| Geometry only    | 74.8%           | Uses only geometric descriptor/prototypes |
| Hybrid model     | 94.1%           | Trainable fusion network with scaled features |

> These numbers are an example reporting format. Recompute on your FERET/LFW split using the scripts under `scripts/`.
