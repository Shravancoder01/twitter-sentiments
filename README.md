# Twitter Sentiment Analysis 🐦📊

**Advanced Sentiment Classification System with Interactive Web Interface**

A comprehensive machine learning project that analyzes Twitter data to classify sentiments as positive, negative, or neutral. This project includes exploratory data analysis, multiple ML model comparison, and a professional Streamlit web application for real-time sentiment prediction.

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.6%2B-green.svg)](https://www.nltk.org/)

## 🎯 Project Overview

This project implements a complete sentiment analysis pipeline from data preprocessing to deployment. It analyzes Twitter sentiment data to understand public opinion and emotional tone behind social media posts. The system uses advanced natural language processing techniques and machine learning algorithms to achieve high-accuracy sentiment classification.

### Key Features

- **Multi-Model Comparison**: Evaluates 10+ machine learning algorithms including Naive Bayes, Logistic Regression, SVM, Random Forest, and ensemble methods
- **Advanced Text Processing**: Implements comprehensive text preprocessing with NLTK including tokenization, stemming, and stopword removal
- **Interactive Web Interface**: Professional Streamlit application with real-time sentiment analysis and visualization
- **Comprehensive EDA**: Detailed exploratory data analysis with word clouds, distribution plots, and correlation analysis
- **Model Persistence**: Trained models and vectorizers are saved for production deployment
- **Performance Metrics**: Detailed accuracy, precision, and confusion matrix analysis

## 📊 Dataset Information

The project uses Twitter sentiment data with the following structure:
- **Text**: Raw tweet content
- **Category**: Sentiment labels (-1: Negative, 0: Neutral, 1: Positive)
- **Size**: Large-scale dataset with thousands of tweets
- **Preprocessing**: Clean text data ready for analysis

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shravancoder01/twitter-sentiments.git
   cd twitter-sentiments/twitter_sentiment_project_v2
   ```

2. **Create virtual environment (recommended)**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('vader_lexicon')
   ```

### Quick Start

1. **Train the models**
   ```bash
   python modal.py
   ```
   This will:
   - Load and preprocess the dataset
   - Perform exploratory data analysis
   - Train multiple ML models
   - Compare model performance
   - Save the best model and vectorizer

2. **Run the web application**
   ```bash
   streamlit run app.py
   ```
   Navigate to `http://localhost:8501` in your browser

## 📁 Project Structure

```
twitter_sentiment_project_v2/
│
├── data/
│   └── Twitter_Data.csv          # Main dataset
│
├── models/
│   ├── model.pkl                 # Trained model (StackingClassifier)
│   └── vectorizer.pkl            # TF-IDF vectorizer
│
├── app.py                        # Streamlit web application
├── modal.py                      # Main training script
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🔬 Machine Learning Pipeline

### 1. Data Preprocessing
- **Text Cleaning**: Removes URLs, mentions, hashtags, and special characters
- **Tokenization**: Splits text into individual words using NLTK
- **Stopword Removal**: Eliminates common English stopwords
- **Stemming**: Reduces words to their root form using Porter Stemmer
- **Feature Engineering**: Extracts character count, word count, and sentence count

### 2. Feature Extraction
- **TF-IDF Vectorization**: Converts text to numerical features
- **N-gram Analysis**: Captures word relationships and context
- **Dimensionality**: Optimized feature space for efficient processing

### 3. Model Training & Evaluation

The system evaluates multiple algorithms:

| Algorithm | Type | Strengths |
|-----------|------|-----------|
| MultinomialNB | Probabilistic | Fast, works well with text data |
| BernoulliNB | Probabilistic | Binary features, spam detection |
| Logistic Regression | Linear | Interpretable, baseline model |
| SVM | Kernel-based | Effective in high dimensions |
| Random Forest | Ensemble | Robust, handles overfitting |
| Stacking Classifier | Meta-ensemble | Combines multiple models |

### 4. Model Selection
- **Cross-validation**: Ensures robust performance estimates
- **Metrics**: Accuracy, precision, recall, F1-score
- **Final Model**: StackingClassifier (combines MultinomialNB + LogisticRegression)

## 🖥️ Web Application Features

### User Interface
- **Modern Design**: Professional CSS styling with responsive layout
- **Real-time Analysis**: Instant sentiment prediction
- **Interactive Visualizations**: Plotly charts for probability distribution
- **Session Statistics**: Track analysis count and characters processed
- **Example Texts**: Quick-start examples for each sentiment class

### Analysis Capabilities
- **Confidence Scoring**: Shows model certainty percentage
- **Detailed Breakdown**: Probability scores for all sentiment classes
- **Processing Metrics**: Word count, character count, processing time
- **Visual Feedback**: Color-coded results and progress animations

## 📈 Performance Metrics

### Model Comparison Results
- **Best Accuracy**: ~85-90% (StackingClassifier)
- **Processing Speed**: <1 second per prediction
- **Memory Usage**: Optimized for production deployment
- **Scalability**: Handles batch processing efficiently

### Visualization Features
- **Word Clouds**: Visual representation of frequent words by sentiment
- **Distribution Plots**: Character, word, and sentence count analysis
- **Correlation Heatmaps**: Feature relationship analysis
- **Performance Charts**: Model comparison and accuracy metrics

## 🛠️ Usage Examples

### Command Line Usage
```python
from sklearn.externals import joblib
import pickle

# Load trained models
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict sentiment
text = "I love this product! It's amazing!"
processed_text = transform_text(text)  # Your preprocessing function
features = vectorizer.transform([processed_text])
prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0]

print(f"Sentiment: {prediction}")
print(f"Confidence: {max(probability):.2%}")
```

### Web Interface Usage
1. Enter text in the input area
2. Click "Analyze Sentiment" button
3. View results with confidence scores
4. Explore detailed probability breakdown
5. Use example texts for quick testing

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom model paths
MODEL_PATH=models/model.pkl
VECTORIZER_PATH=models/vectorizer.pkl

# Streamlit configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

### Custom Settings
Modify `app.py` for:
- Custom styling and themes
- Additional visualization options
- Performance optimizations
- Authentication features

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit changes** (`git commit -m 'Add AmazingFeature'`)
4. **Push to branch** (`git push origin feature/AmazingFeature`)
5. **Open Pull Request**

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to functions
- Include unit tests for new features
- Update documentation as needed

## 📋 Requirements

### Core Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
nltk>=3.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
wordcloud>=1.8.0
streamlit>=1.0.0
plotly>=5.0.0
```

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB free space
- **CPU**: Any modern processor
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```

### Cloud Deployment Options
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Container-based deployment
- **AWS EC2**: Full control over environment
- **Google Cloud Run**: Serverless container deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🐛 Troubleshooting

### Common Issues

**1. NLTK Data Missing**
```python
import nltk
nltk.download('all')  # Downloads all NLTK data
```

**2. Memory Issues with Large Dataset**
```python
# Use data sampling for development
df_sample = df.sample(n=10000, random_state=42)
```

**3. Model Loading Errors**
- Ensure models are trained before running the app
- Check file paths in `models/` directory
- Verify pickle file compatibility

**4. Streamlit Port Issues**
```bash
streamlit run app.py --server.port 8502  # Use different port
```

## 📊 Results & Insights

### Key Findings
- **Positive Sentiment**: Often contains words like "love", "great", "awesome"
- **Negative Sentiment**: Frequently includes "hate", "terrible", "worst"
- **Neutral Sentiment**: Factual statements without emotional indicators
- **Model Performance**: Ensemble methods outperform individual classifiers

### Business Applications
- **Brand Monitoring**: Track customer sentiment about products/services
- **Market Research**: Understand public opinion on topics
- **Customer Support**: Prioritize negative feedback for quick response
- **Social Media Strategy**: Optimize content based on sentiment trends

## 🔍 Future Enhancements

### Planned Features
- [ ] **Multi-language Support**: Extend beyond English tweets
- [ ] **Real-time Twitter Integration**: Live tweet streaming and analysis
- [ ] **Advanced Visualizations**: Interactive dashboards and reports
- [ ] **Model Improvements**: Deep learning models (BERT, RoBERTa)
- [ ] **API Development**: RESTful API for integration with other systems
- [ ] **Batch Processing**: Handle large-scale data processing
- [ ] **A/B Testing**: Compare different model versions
- [ ] **User Authentication**: Secure access and user management

### Technical Improvements
- **Performance Optimization**: Faster inference and processing
- **Memory Efficiency**: Reduced resource consumption
- **Error Handling**: Comprehensive error management
- **Logging System**: Detailed application logging
- **Testing Suite**: Unit and integration tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Shravan Chafekar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 👨‍💻 Author

**Shravan Chafekar** (Shravancoder01)
- GitHub: [@Shravancoder01](https://github.com/Shravancoder01)
- LinkedIn: [Connect with me](https://linkedin.com/in/shravanchafekar)
- Email: [Contact](mailto:shravan.chafekar@email.com)

## 🙏 Acknowledgments

- **NLTK Team**: For comprehensive natural language processing tools
- **Scikit-learn Community**: For robust machine learning algorithms
- **Streamlit**: For the amazing web application framework
- **Open Source Community**: For valuable feedback and contributions
- **Dataset Contributors**: For providing quality Twitter sentiment data

## 📚 References & Citations

1. Bird, Steven, Edward Loper and Ewan Klein (2009), *Natural Language Processing with Python*. O'Reilly Media Inc.
2. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
3. Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.
4. Liu, B. (2012). *Sentiment Analysis and Opinion Mining*. Morgan & Claypool Publishers.

## 🔗 Useful Links

- [Streamlit Documentation](https://docs.streamlit.io/)
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Twitter API Documentation](https://developer.twitter.com/en/docs)
- [Sentiment Analysis Research Papers](https://scholar.google.com/scholar?q=sentiment+analysis+twitter)

---

⭐ **If you found this project helpful, please give it a star!** ⭐

*Built with ❤️ for the data science and machine learning community*