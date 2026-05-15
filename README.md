# Omni-Guard : Multi-Modal Compliance & IPR Auditor

Omni-Guard is an advanced AI-driven auditing suite designed to automate the detection of regulatory violations and Intellectual Property (IP) theft in digital media. By combining Computer Vision (OCR), Speech-to-Text (ASR), and Large Language Models (LLMs), it provides a comprehensive 360-degree analysis of video and image content.

## 🚀 Overview

Traditional moderation tools often fail to capture the context of a video. Omni-Guard solves this by "hearing" the audio and "seeing" the visual text simultaneously. It cross-references this data against global databases to ensure that content adheres to legal standards regarding politics, regulated substances, and medical claims.

## Features

- Multi-modal extraction: ASR (audio) + OCR (frames) + LLM reasoning
- Plagiarism/structure similarity scoring against a BigQuery reference corpus
- Political-communication detection (disclaimer/authorization checks)
- Regulated-substance promotion detection (alcohol, tobacco, vaping)
- Medical and weight-loss claim verification and disclaimer checks
- Automated contextual compliance reporting
- Real-time multimedia audit workflow
- Multi-domain regulatory risk assessment
- Cross-modal verification using audio and visual evidence

## Core Audit Domains

### 1. Subtitle & Script Theft (IPR)

- Method: OCR + ASR + TF-IDF / Cosine similarity
- Generates a Plagiarism Risk score by comparing extracted narrative content against a BigQuery reference corpus.

### 2. Political Compliance

- Method: Zero-shot classification
- Detects political intent, campaign messaging, and absence of mandatory disclaimers such as “Paid for by” or “Authorized by”.

### 3. Regulated Substances

- Method: Multi-modal keyword and intent mapping
- Identifies covert promotions of alcohol, tobacco, and vaping products using OCR and ASR correlation.

### 4. Medical & Weight-Loss Claims

- Method: Semantic analysis
- Detects extraordinary or misleading medical claims and flags missing medical disclaimers.

## Workflow Logic

1. User uploads a video (`.mp4`, `.mov`) or image file.
2. Whisper converts audio into a searchable transcript.
3. EasyOCR extracts visible text from sampled video frames.
4. Relevant records are retrieved from the configured BigQuery dataset.
5. TF-IDF + Cosine Similarity computes plagiarism and structural similarity scores.
6. Gemini performs semantic reasoning and contextual compliance analysis.
7. A final audit report is generated with domain-wise risk classifications.

## Architecture & Tech Stack

### AI & Machine Learning

- **LLM:** Gemini (semantic reasoning and contextual review)
- **ASR:** OpenAI Whisper (Tiny) for speech-to-text
- **OCR:** EasyOCR for on-screen text extraction
- **Similarity Engine:** scikit-learn (TF-IDF + Cosine similarity)

### Infrastructure & Processing

- **Video Processing:** MoviePy for frame sampling and audio extraction
- **Database:** Google BigQuery (reference corpus such as patents-public-data)
- **Execution Environment:** Google Colab (recommended; use T4 GPU for acceleration)

## Quickstart (Google Colab)

1. Open the notebook `prism-eipr.ipynb` in Google Colab.
2. Set runtime to GPU (T4) via Runtime → Change runtime type.
3. Upload `credentials.json` for GCP access to the Colab workspace (or mount GCS as appropriate).
4. Add your LLM/API keys to Colab secrets (e.g., `GEMINI_API_KEY`).
5. Run cells in order to ingest a video and produce an audit report.

## Configuration

- **GEMINI_API_KEY:** Store in Colab Secrets or as an environment variable.
- **GCP Credentials:** Place `credentials.json` in the working directory or configure application default credentials.
- **BigQuery Dataset/Table:** Configure the notebook variable pointing to the reference corpus (example: `legal_data.patent_corpus`).

## Example Output

The notebook produces a human-readable compliance report, for example:

```plaintext
📊 IPR PLAGIARISM RISK: 12.7%

⚖️ COMPLIANCE SUMMARY:
- POLITICS: Non-compliant (political intent detected; missing disclaimers)
- SUBSTANCES: Compliant (no promotion detected)
- WEIGHT LOSS: HIGH RISK (unsubstantiated claims; missing medical disclaimer)
```

## Notes & Requirements

- Recommended: Run in Google Colab with a T4 GPU for reasonable performance.
- BigQuery credentials are required to query the reference corpus.
- CUDA acceleration significantly improves Whisper and OCR performance.
- Ensure all required Python dependencies are installed before execution.
- This project assumes lawful use and appropriate access to third-party datasets.

## Dependencies

```bash
pip install openai-whisper easyocr moviepy scikit-learn google-cloud-bigquery opencv-python
```

## Supported Input Formats

### Video
- `.mp4`
- `.mov`
- `.avi`

### Images
- `.png`
- `.jpg`
- `.jpeg`

## Files

- `prism-eipr.ipynb` — Primary analysis notebook for ingestion and auditing
- `credentials.json` — Google Cloud credentials for BigQuery access
- `README.md` — Project documentation

## Future Enhancements

- Live-stream moderation support
- Multi-language compliance auditing
- Advanced logo and brand detection
- Deepfake and synthetic media analysis
- Dashboard integration for enterprise monitoring

## License & Contact

This repository contains prototype code and notes. Add a license file (e.g., `LICENSE`) to specify terms. For questions or collaboration, open an issue or contact the author.