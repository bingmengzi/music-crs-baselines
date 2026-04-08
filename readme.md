# Music Conversational Recommendation Challenge — Baselines

Welcome to the official baseline repository for the **Music Conversational Recommendation Challenge (Music-CRS)**, a global research competition bridging Natural Language Processing and Recommender Systems. Participants build systems that engage in multi-turn conversations about music, recommend relevant tracks, and generate natural language responses.

The challenge uses **TalkPlayData-Challenge**, a large-scale synthetic multi-turn dialogue dataset grounded in real music listening histories, with pre-extracted multimodal track and user embeddings provided.

Dataset splits: **Train**, **Development**, **Blind A**, **Blind B**

---

## Organizing Committee

| Name | Affiliation |
|---|---|
| Seungheon Doh | KAIST, South Korea |
| Sergio Oramas | Pandora/SiriusXM |
| Bruno Sguerra | Deezer Research |
| Abhinav Bohra | Amazon |
| Claudio Pomo | Politecnico di Bari, Italy |
| Francesco Barile | Maastricht University, Netherlands |

---

## Timeline

| Date | Milestone |
|---|---|
| 31 March 2026 | Website online |
| 10 April 2026 | Start RecSys Challenge — Release dataset (Train, Development, Blind A) |
| 15 April 2026 | Submission System Open — Leaderboard live (with Blind A dataset) |
| 15 June 2026 | Blind Dataset B released — Submission system activated for Blind B |
| 30 June 2026 | End RecSys Challenge |
| 6 July 2026 | Final Leaderboard & Winners — EasyChair open for submissions |
| 9 July 2026 | Upload code of final predictions |
| 20 July 2026 | Paper Submission Due |
| 3 August 2026 | Paper Acceptance Notifications |
| 10 August 2026 | Camera-Ready Papers |
| September 2026 | RecSys Challenge Workshop at ACM RecSys 2026 |

---

## Challenge Overview

Build a conversational AI that can:
- Understand user music preferences through natural multi-turn dialogue
- Recommend relevant tracks from a large music catalog
- Generate engaging, personalized natural language responses about music

---

## Baseline System

The system operates on a **two-stage pipeline**:
1. **RecSys** — Retrieve candidate tracks matching user preferences
2. **LLM** — Generate a natural language response explaining the recommendations

### Core Components

| Component | Description | Module |
|---|---|---|
| LLM | Generates natural language responses (Llama-3.2-1B-Instruct) | `mcrs/lm_modules/` |
| RecSys | Retrieves relevant tracks via BM25 (sparse) or BERT (dense) | `mcrs/retrieval_modules/` |
| User DB | Stores user profiles (user_id, age, gender, country) | `mcrs/db_user/user_profile.py` |
| Item DB | Contains track metadata (name, artist, album, tags, release date) | `mcrs/db_item/music_catalog.py` |

---

## Challenge Resources

- **Dataset collection**: [TalkPlayData-Challenge](https://huggingface.co/collections/talkpl-ai/talkplay-data-challenge)
- **Conversation Dataset**: [TalkPlayData-Challenge-Dataset](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Dataset)
- **Track Metadata**: [TalkPlayData-Challenge-Track-Metadata](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Track-Metadata)
- **User Profiles**: [TalkPlayData-Challenge-User-Metadata](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-User-Metadata)
- **Blind A Dataset**: [TalkPlayData-Challenge-Blind-A](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Blind-A)
- **Blind B Dataset**: Will be uploaded @ 15 Jun

---

## Quick Start

### Installation

```bash
uv venv .venv --python=3.10
source .venv/bin/activate
uv pip install -e .
uv pip install flash-attn --no-build-isolation # for fast llm inference
```

### Run Inference on the Development Set

**⚠️ Note: During inference, the recommender system must always retrieve candidates from the entire track catalog. Do not filter, subset, or restrict tracks using `track_split_types` or any other mechanism!**
For BM25/BERT baselines, your config must include:
```yaml
track_split_types:
  - "all_tracks"
```
If you do not use `all_tracks`, your evaluation may be considered invalid.

- Always use `all_tracks` for every experiment and submission.
- Do **not** preprocess, filter, or use only a subset of tracks during inference.



```bash
# BM25 baseline
python run_inference_devset.py --tid llama1b_bm25_devset --batch_size 16

# BERT baseline
python run_inference_devset.py --tid llama1b_bert_devset --batch_size 16
```

Results are saved to `exp/inference/{tid}.json`.

### Run Inference on Blind Sets (for submission)

```bash
# BM25 baseline
python run_inference_blindset.py --tid llama1b_bm25_blindset_A --batch_size 16

# BERT baseline
python run_inference_blindset.py --tid llama1b_bert_blindset_A --batch_size 16
```

---

## Custom Configuration

Create a config file in `config/`:

```yaml
# config/my_model.yaml
lm_type: "meta-llama/Llama-3.2-1B-Instruct"
retrieval_type: "qwen_embedding"
item_db_name: "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
user_db_name: "talkpl-ai/TalkPlayData-Challenge-User-Metadata"
split_types:
  - "test_warm"
  - "test_cold"
corpus_types:
  - "track_name"
  - "artist_name"
  - "album_name"
  - "tag_list"
cache_dir: "./cache"
device: "cuda"
attn_implementation: "flash_attention_2"
```

Then run with your config:

```bash
python run_inference_devset.py --tid my_model
```

---

## Evaluation

For evaluation, please refer to: https://github.com/nlp4musa/music-crs-evaluator

---

## Tips & Extensions

See `./tips/` for advanced techniques. Some directions to explore:

- **Improve Item Representation** — Add audio features or use stronger embedding models
- **Add a Reranker Module** — Implement two-stage ranking with LLM or embedding-based rerankers
- **Generative Retrieval** — Use semantic IDs for end-to-end track generation

---

Good luck with the challenge!
