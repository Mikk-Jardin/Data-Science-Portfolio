# Name
## SkimLit NLP Deep Learning Model
Based on the paper "PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts" (https://arxiv.org/pdf/1710.06071.pdf)
Project done inline with course TensorFlow Developer Certificate 2021: Zero to Mastery

# Description
Deep learning NLP model used for sequential sentence classification of medical paper abstracts.
Classifies the sentences of a medical paper's abstract in to the following classes:  background, objective, method, result, or conclusion.
Makes use of transfer learning (Universal Sentence Encoder: https://tfhub.dev/google/universal-sentence-encoder/4).
Includes:
- Data batching and prefetching techniques to optimize model training.
- Feature Engineering
- Model experimentation and validation

# Model Visuals
**Hybrid Character and Token-level Embedding**: Model that makes use of token and character-level embeddings.
<img width="503" alt="Screen Shot 2021-07-16 at 10 31 29 AM" src="https://user-images.githubusercontent.com/66507226/125882991-6e4849fc-336a-4821-8170-4052f562b314.png">

**Tribrid Character and Token-level + Engineered Features Embedding**: Model that makes use of token and character-level embeddings plus engineered features of text (line_number and total lines of abstract)
<img width="641" alt="Screen Shot 2021-07-16 at 10 31 44 AM" src="https://user-images.githubusercontent.com/66507226/125883000-8f87a2ae-e27d-4e65-8a6d-9d0419f54af7.png">
