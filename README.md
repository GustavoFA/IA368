# IA368 - Advanced Deep Learning: LLMs and Intelligent Agents in Practice 

This course covers neural network architectures applied to Natural Language Processing (NLP), with a focus on modern Large Language Models (LLMs). Topics include text representations (bag-of-words, word embeddings), sequence-to-sequence models, attention mechanisms, and the Transformer architecture. The course also covers pre-training and fine-tuning strategies (e.g., BERT, instruction tuning), efficient training techniques, and post-training inference strategies. Advanced topics include prompt engineering, ReAct (Reasoning + Acting), Retrieval-Augmented Generation (RAG), and the development of agents and multi-agent systems. A distinctive feature of the course is the intensive use of AI-assisted programming tools to accelerate development and experimentation. Implementation is done in Python and PyTorch, and the course concludes with a practical final project integrating all major concepts.

## Notebooks

### Language Model — Bengio (MLP + Embeddings)

Implementation of a neural language model inspired by Bengio et al. (2003) for next-word prediction, trained on a Portuguese literary corpus (Machado de Assis). The model uses a vocabulary of 20,000 words, word embeddings (dim=128), a Tanh hidden layer, and a linear output layer over the vocabulary. Training used a 5-word sliding context window, AdamW optimizer, cross-entropy loss, and ran for 10 epochs. The model achieved a final validation perplexity of ~146, below the target of 200, and is capable of generating coherent Portuguese text from a given seed context.



