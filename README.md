# keras-custom-attention-sms-spam-sms-spam-detection-transformer
Lightweight Transformer encoder with custom Multi-Head Self-Attention built from scratch in Keras 3 for binary SMS spam detection with class-imbalance handling (98% test accuracy).
• Overview: Binary text classification model trained on the UCI SMS Spam Collection (5,171 unique messages after deduplication).
• Preprocessing: Custom NLTK pipeline featuring WordNet lemmatization and special token normalization (<num> for numbers/currencies and <caps> for all-caps words).
• Architecture & Training: Custom TokenPositionEmbedding and MultiHeadSelfAttention layers built from scratch without high-level Transformer wrappers, trained using balanced class weights to handle the 87% ham / 13% spam skew.
• Results: Achieves 98% overall test accuracy, 0.95 recall on Spam, and 0.99 F1-score on Ham.
• Tech Stack: Python, Keras 3, TensorFlow, NLTK, Scikit-learn.
