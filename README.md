# IA368 - Advanced Deep Learning: LLMs and Intelligent Agents in Practice 

This course covers neural network architectures applied to Natural Language Processing (NLP), with a focus on modern Large Language Models (LLMs). Topics include text representations (bag-of-words, word embeddings), sequence-to-sequence models, attention mechanisms, and the Transformer architecture. The course also covers pre-training and fine-tuning strategies (e.g., BERT, instruction tuning), efficient training techniques, and post-training inference strategies. Advanced topics include prompt engineering, ReAct (Reasoning + Acting), Retrieval-Augmented Generation (RAG), and the development of agents and multi-agent systems. A distinctive feature of the course is the intensive use of AI-assisted programming tools to accelerate development and experimentation. Implementation is done in Python and PyTorch, and the course concludes with a practical final project integrating all major concepts.

## Notebooks

### Language Model — Bengio (MLP + Embeddings)

Implementation of a neural language model inspired by Bengio et al. (2003) for next-word prediction. The model uses word embeddings followed by a non-linear hidden layer and a softmax output layer over the vocabulary. Evaluated by perplexity (target < 200) and capable of generating text from a given word context.

