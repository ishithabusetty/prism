Omni-Guard: Multi-Modal Compliance & IPR Auditor
Omni-Guard is an advanced AI-driven auditing suite designed to automate the detection of regulatory violations and Intellectual Property (IP) theft in digital media. By combining Computer Vision (OCR), Speech-to-Text (ASR), and Large Language Models (LLMs), it provides a comprehensive 360-degree analysis of video and image content.

🚀 Overview
Traditional moderation tools often fail to capture the context of a video. Omni-Guard solves this by "hearing" the audio and "seeing" the visual text simultaneously. It cross-references this data against global databases to ensure that content adheres to legal standards regarding politics, regulated substances, and medical claims.

🛠 Tech Stack
AI & Machine Learning
LLM (The Brain): Gemini 2.5 Flash — Handles semantic reasoning, intent detection, and multi-domain regulatory audits.

ASR (Hearing): OpenAI Whisper (Tiny) — Converts spoken dialogue into searchable text.

OCR (Seeing): EasyOCR — Extracts on-screen text, brand names, and hidden disclaimers from video frames.

Similarity Engine: Scikit-Learn — Uses TF-IDF Vectorization and Cosine Similarity for mathematical plagiarism detection.

Data & Infrastructure
Database: Google BigQuery — Interfaces with global patent and publication datasets (e.g., patents-public-data) to act as a reference corpus for IPR.

Video Processing: MoviePy — Handles frame-by-frame sampling and audio stream isolation.

Environment: Google Colab — Optimized for T4 GPU acceleration to ensure rapid multi-modal processing.

📋 Core Audit Domains
1. Subtitle & Script Theft (IPR)
Method: OCR + ASR + TF-IDF Cosine Similarity.

Logic: The system extracts the combined narrative (what is said + what is shown) and compares it against the BigQuery reference corpus. It generates a Plagiarism Risk Percentage to identify if the script structure or technical content is stolen from a published work.

2. Political Compliance
Method: Zero-Shot Classification.

Logic: Detects political intent or electioneering. It specifically scans for mandatory "Paid for by" or "Authorized by" disclaimers. If political intent is found without these markers, it flags a high-risk violation.

3. Regulated Substances
Method: Multi-Modal Keyword & Intent Mapping.

Logic: Identifies the promotion of Alcohol, Tobacco, or Vaping. It cross-references visual branding (OCR) with spoken promotion (ASR) to catch "covert" advertising.

4. Medical & Weight Loss Scams
Method: Semantic Analysis.

Logic: Analyzes claims of "instant" or "miracle" results. It enforces compliance by checking for the absence of mandatory medical disclaimers like "Results not typical" or "Consult a healthcare professional."

🔄 Workflow Logic
Ingestion: User uploads a video (.mp4, .mov) or image to the notebook.

Multimodal Extraction: * Whisper transcribes the audio into a transcript.

EasyOCR samples video frames (at 50% duration) to capture on-screen text.

Reference Retrieval: The system pulls up to 200 relevant records from the specified BigQuery table (e.g., legal_data.patent_corpus).

IPR Scoring: A mathematical similarity score is calculated between the extracted video text and the BigQuery corpus.

LLM Audit: The Gemini 2.5 Flash model performs a "Contextual Review" to generate a detailed compliance report, flagging specific violations and reasoning.

⚙️ Setup & Configuration
API Keys: * Store your GEMINI_API_KEY in the Colab Secrets tab.

Obtain your key from the Google AI Studio (as seen in your provided documentation).

GCP Credentials: * Upload your credentials.json file to the /content/ directory to enable BigQuery access.

Hardware: * Ensure the notebook is set to T4 GPU (Runtime > Change runtime type) to enable CUDA acceleration for Whisper and EasyOCR.

📈 Example Output
Plaintext
📊 IPR PLAGIARISM RISK: 0.0%
⚖️ COMPLIANCE REPORT:
- POLITICS: Compliant (No intent found)
- SUBSTANCES: Compliant (No promotion found)
- WEIGHT LOSS: HIGH RISK (Claims 'dramatic change' without medical disclaimer)