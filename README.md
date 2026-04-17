# IA368 - Advanced Deep Learning: LLMs and Intelligent Agents in Practice 

Professor: Profa. Dr. Roberto de Alencar Lotufo
Semester: 2nd Semester 2025
Institution: FEEC — UNICAMP

This course covers neural network architectures applied to Natural Language Processing (NLP), with a focus on modern Large Language Models (LLMs). Topics include text representations (bag-of-words, word embeddings), sequence-to-sequence models, attention mechanisms, and the Transformer architecture. The course also covers pre-training and fine-tuning strategies (e.g., BERT, instruction tuning), efficient training techniques, and post-training inference strategies. Advanced topics include prompt engineering, ReAct (Reasoning + Acting), Retrieval-Augmented Generation (RAG), and the development of agents and multi-agent systems. A distinctive feature of the course is the intensive use of AI-assisted programming tools to accelerate development and experimentation. Implementation is done in Python and PyTorch, and the course concludes with a practical final project integrating all major concepts.

## Notebooks

### 1 - Language Model - Bengio (MLP + Embeddings)

Implementation of a neural language model inspired by Bengio et al. (2003) for next-word prediction, trained on a Portuguese literary corpus (Machado de Assis). The model uses a vocabulary of 20,000 words, word embeddings (dim=128), a Tanh hidden layer, and a linear output layer over the vocabulary. Training used a 5-word sliding context window, AdamW optimizer, cross-entropy loss, and ran for 10 epochs. The model achieved a final validation perplexity of ~146, below the target of 200, and is capable of generating coherent Portuguese text from a given seed context.

### 2 - Language Model - Self-Attention

This activity implements a neural language model using self-attention to predict the next word based on previous tokens. The model includes positional embeddings, linear projections (WQ, WK, WV, WO), and a 2-layer feed-forward network. Two versions of the self-attention mechanism were developed: a loop-based (intuitive but inefficient) and a vectorized (efficient) implementation, with an assertion confirming identical outputs. Training was performed using the matrix-based version, showing that the model can learn contextual dependencies effectively while maintaining computational efficiency.

### 3 - BERT - Bidirectional Encoder Representations from Transformers

This activity implements a language model to predict the next token and evaluate performance using perplexity on a dataset from Machado de Assis, leveraging embeddings from a pre-trained BERT model. The pipeline uses tokenized integer inputs, where contextual representations are extracted from the last token embedding of BERT and passed to a lightweight MLP with output size equal to the vocabulary. A custom training loop was developed without high-level training frameworks, and experiments were conducted varying context size and whether BERT parameters were frozen or fine-tuned. The results show that increasing context improves predictive performance at the cost of higher computational load, while freezing BERT reduces training time with a slight drop in accuracy, achieving reasonable perplexity values given the constrained MLP capacity.

### 4 - Language Model - Self-Attention and Causal Masking

This activity extends the previous language model by incorporating causal masking into a self-attention architecture to ensure that predictions depend only on past tokens, following the paradigm used in models like GPT. Using a matrix-based implementation of self-attention, a causal mask was added to prevent access to future tokens during training. The tokenizer and dataset were adapted to include ```<sos>``` and ```<eos>``` tokens, and to generate input–target pairs with a one-step left shift, enabling next-token prediction over sequences with a fixed context size. The model was further enhanced with multi-head attention, allowing it to capture different representation subspaces. After validating the data pipeline and masking mechanism, the model was trained on Machado de Assis texts, demonstrating the ability to learn coherent sequential dependencies while respecting the autoregressive constraint imposed by the causal mask

### 5 - Language Model with Self-Attention and Causal Masking + LoRA

This activity explores parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation) to adapt a pre-trained language model with reduced computational cost. Instead of updating all model parameters, low-rank decomposition matrices are injected into specific layers, allowing the model to learn task-specific representations while keeping most weights frozen. A custom training loop was implemented to train only the LoRA parameters, significantly reducing memory usage and training time. Experiments demonstrate that LoRA achieves competitive performance compared to full fine-tuning, highlighting its effectiveness for adapting large models under limited resource constraints.

### 6 - CLIP - Multimodal Search via Embedding

This activity implemented a Multimodal Search system by aligning text and image embeddings from the STL10 dataset into a shared latent space. The architecture utilized frozen pre-trained models—BERT for text and EfficientNet-b0 for images—with added trainable linear layers for fine-tuning. By comparing MSE, Contrastive, and SigLIP loss functions, the results showed that contrastive strategies significantly improved retrieval performance by effectively separating negative pairs. The final system achieved successful text-to-image retrieval, outperforming a standard zero-shot CLIP test in this specific domain.

### 7 - RAG - Retrieval-Augmented Generation

This activity focused on developing a Retrieval-Augmented Generation (RAG) system using the IIRC dataset and the Visconde paper methodology. The pipeline consisted of four main stages: segmentation of context articles into smaller chunks; dense retrieval using sentence-transformer embeddings and a FAISS index for efficient similarity searching; generation of answers via the GPT-4o-mini model using a custom prompt to integrate retrieved contexts; and evaluation using the F1-bag-of-words metric. Results indicated that the system was particularly effective at handling "span" type questions, achieving high precision and solid recall by successfully grounding the LLM's responses in the retrieved text and significantly reducing hallucinations.

### 8 - Prompt Engineering 

This activity explored Prompt Engineering techniques for Sentiment Analysis using the IMDb dataset and the gpt-4o-mini model. The objective was to compare the effectiveness of three distinct prompting strategies—Zero-Shot, Few-Shot, and Chain-of-Thought (CoT)—in classifying movie reviews as positive or negative. By systematically testing these methods on a 1,000-review subset, the results demonstrated that while Zero-Shot provides a quick baseline, Few-Shot and Chain-of-Thought significantly enhance the model's accuracy and reasoning capabilities, particularly for nuanced or sarcastic reviews. The final implementation successfully automated the classification and evaluation process, using the F1-score and accuracy metrics to confirm that structured logical reasoning (CoT) leads to the most robust and reliable sentiment detection.