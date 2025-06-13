

### 1. Foundation Models
- Large Deep Learning Neural Networks trained on massive datasets. 
- data scientists use a foundation model as a starting point to develop ML models.
- before FMs: traditional ML models for specific tasks such as sentiment analysis, classifying images, forecasting trends. 
- Example Use Cases:
    - Customer Support
    - Language Translation
    - Content Generation
    - Copy Writing
    - Image Classification
    - High-resolution image creation and editing
    - Document Extraction
    - Robotics
    - Healthcare
    - Autonomous Vehicles


### 2. BERT
Released in 2018, Bidirectional Encoder Representations from Transformers (BERT) was one of the first foundation models. BERT is a bidirectional model that analyzes the context of a complete sequence then makes a prediction. It was trained on a plain text corpus and Wikipedia using 3.3 billion tokens (words) almost 16 GB training dataset and 340 million parameters. BERT can answer questions, predict sentences, and translate texts. According to OpenAI, the computational power required for foundation modeling has doubled every 3.4 months since 2012. 

### 3. GPT  
The Generative Pre-trained Transformer (GPT) model was developed by OpenAI in 2018. It uses a 12-layer transformer decoder with a self-attention mechanism. And it was trained on the BookCorpus dataset, which holds over 11,000 free novels. `A notable feature of GPT-1 is the ability to do zero-shot learning`.


GPT-2 released in 2019. OpenAI trained it using 1.5 billion parameters (compared to the 117 million parameters used on GPT-1). GPT-3 has a 96-layer neural network and 175 billion parameters and is trained using the 500-billion-word Common Crawl dataset. The popular ChatGPT chatbot is based on GPT-3.5. And GPT-4, the latest version, launched in late 2022 and successfully passed the Uniform Bar Examination with a score of 297 (76%).


### 4. How Foundational Models Work
Models are based on complex neural networks including generative adversarial networks (GANs), transformers, and variational encoders. In general, an FM uses learned patterns and relationships to predict the next item in a sequence. For example, with image generation, the model analyzes the image and creates a sharper, more clearly defined version of the image. Similarly, with text, the model predicts the next word in a string of text based on the previous words and its context. It then selects the next word using probability distribution techniques.


Foundation models use self-supervised learning to create labels from input data. This means no one has instructed or trained the model with labeled training data sets. This feature separates LLMs from previous ML architectures, which use supervised or unsupervised learning.

### 5. Examples of Foundation Models
- BERT
- GPT
- Amazon Nova
- A121 Jurassic