# Biomedical Question Answering with PubMedQA

This project explores biomedical question answering using the **PubMedQA** dataset and various modeling approaches, from classical NLP to large language models. The goal is to both **classify yes/no answers** and **generate long answers** for biomedical questions.

## 📚 Dataset

We use the [PubMedQA dataset](https://pubmedqa.github.io), which includes:

- **PQA-L**: ~1,000 labeled examples  
- **PQA-A**: ~211,000 artificial labeled examples  
- **PQA-U**: ~61,000 unlabeled examples  

Each sample includes a **question**, **context**, **long answer**, and a **final answer** ("yes", "no", or "maybe").

## 🔍 Data Preprocessing & Exploration

Our first step involved preprocessing the text:
- Lowercasing
- Removing punctuation and stopwords
- Tokenization
- Lemmatization

We then conducted an initial analysis:
- **Text statistics**: lengths, vocabulary size
- **Clustering**: using **K-Means** and **MiniBatch K-Means**, visualized via **SVD**
- **Topic identification**: clusters revealed themes like *oncology* and *maternal health*
- **Search engine**: implemented using **TF-IDF** and **BM25**
- **Word embeddings**: trained **Word2Vec** on the dataset, capturing domain-specific semantics

## 🧠 Classification Task

We converted the task into **binary classification** by removing "maybe" samples. Due to computational constraints, we trained models on:
- All of PQA-L
- A class-balanced subset of PQA-A

### Models Compared:

| Model | Preprocessing | Accuracy |
|-------|---------------|----------|
| BiLSTM | Basic pipeline | 72% |
| BiLSTM | BioBERT tokenizer | 83% |
| BioBERT | Fine-tuned | 78% |
| BioBERT | Fully retrained | 87% |
| BioBERT | Transfer learning | **93%** ✅ |

> Confusion matrices confirmed strong, balanced performance across classes.  
> **Conclusion**: Domain-specific models like BioBERT are highly effective for biomedical classification.

## 🤖 Q&A Generation with LLaMA2-7B

We framed the long-answer generation task as a **prompt-based Q&A** with **LLaMA2-7B**. Each prompt included:
- Instruction
- Context (abstract)
- Question
- Expected answer (yes/no + explanation)

### Evaluation:
- **Zero-shot**: Poor results
- **One-shot**: Accuracy ~67%, ROUGE tripled
- **Few-shot (5 examples)**: Accuracy ~88%
- **Fine-tuning (8,000 samples)**: Slight further improvements

> **Takeaway**: Few-shot prompting was nearly as good as fine-tuning — LLMs adapt quickly with minimal examples.

## 🚀 Extensions

### 1. Semi-Supervised Learning on PQA-U
To use the unlabeled set:
- Generated pseudo-labels using the best classifier
- Filtered predictions based on **confidence threshold**
- Fine-tuned on high-confidence samples

✅ **Accuracy improved to 93%**, matching the best fully-supervised model.  
> Demonstrates the power of semi-supervised learning when labeled data is limited.

### 2. Speech-to-Text Chatbot
We built a **multimodal chatbot** that:
- Accepts **text or voice input**
- Uses a **paper abstract as context**
- Generates a **concise answer** via the fine-tuned LLaMA model
- Responds with **synthesized voice**

> A prototype for voice-interactive biomedical assistance.

---

## 🛠 Technologies Used

- Python, PyTorch, HuggingFace Transformers
- Scikit-learn, Gensim, NLTK, SpaCy
- BioBERT, LLaMA2
- TF-IDF, BM25
- Word2Vec
- Streamlit (optional UI)
- SpeechRecognition, gTTS (for chatbot)

## 📈 Results Summary

| Task | Best Accuracy | Notes |
|------|----------------|-------|
| Classification | 93% | BioBERT (transfer learning) |
| Q&A Generation | 88% | LLaMA2 few-shot |
| Semi-Supervised | 93% | High-confidence pseudo-labeling |

## 🤝 Contributors
 MOIANA Laura, FRAGERI Martina, GUAZZI Alessandro, LEI Leonardo, MANTEGAZZA Niccolò

