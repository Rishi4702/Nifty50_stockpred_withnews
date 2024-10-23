# Stock Market Forecasting Through News Sentiment Analysis

### Authors: Rishav

Are emotions driving the market? This project explores how the sentiment behind news headlines can provide valuable insights for stock market predictions. By using state-of-the-art transformer-based models, we've created an index similar to the "Greed and Fear Index," analyzing its influence on predicting stock market trends with greater accuracy.

## Project Overview

Stock market forecasting has always been a puzzle, influenced by numerous factors, including the news. This project seeks to answer a key question: Can a sentiment-based index derived from news headlines enhance the precision of stock market prediction models?

To address this, we analyzed news headlines with transformer-based models like BERT and RoBERTa, generating a "Greed and Fear Index" to capture market sentiment. We then combined this index with historical OHCL (Open, High, Close, Low) data and used an LSTM (Long Short-Term Memory) model to forecast stock market trends.

## Highlights

- **Sentiment-Driven Predictions**: Utilized transformer models (BERT, FinBERT, and RoBERTa) to generate a sentiment index from top news headlines, providing a unique "Greed and Fear" perspective.
- **Model Integration**: Combined this sentiment index with historical market data to assess its effect on stock market predictions using an LSTM model.
- **Comparative Analysis**: Compared the performance of the generated indices using different transformer models in boosting the accuracy of LSTM-based stock prediction.

## Key Findings

- **FinBERT Shines**: FinBERT emerged as the most effective transformer model for capturing sentiment, showing strong prediction reliability when integrated with the LSTM model.
- **Impact of Sentiment Index**: The inclusion of the Greed and Fear index improved prediction accuracy, indicating that market sentiment indeed plays a crucial role in stock movements.
- **Hyperparameter Optimization**: The effectiveness of the LSTM model was significantly influenced by hyperparameter tuning, highlighting the need for careful adjustments to achieve the best results.

## Methodology

1. **Sentiment Index Generation**: Transformer models (BERT, FinBERT, RoBERTa) were fine-tuned to generate a sentiment score from news headlines, forming the Greed and Fear index.
2. **Data Integration**: The Greed and Fear index was integrated with historical OHCL stock data to create a comprehensive dataset for stock market forecasting.
3. **LSTM Model**: A custom LSTM model was optimized through rigorous hyperparameter tuning to effectively use the sentiment data for prediction.

## Results

- **Sentiment Analysis Models**: Among BERT, FinBERT, and RoBERTa, FinBERT showed the most balanced performance, demonstrating a reliable sentiment score that aided in stock forecasting.
- **LSTM Performance**: The custom LSTM model trained with both the OHCL data and sentiment index outperformed traditional OHCL-only models, validating the significance of market sentiment.

## Why This Matters

Understanding the role of emotions in the stock market can lead to more informed investment strategies. This project illustrates how advanced NLP techniques can provide deeper insights into market trends, offering an innovative approach to financial forecasting. By combining news sentiment with technical indicators, we can gain a competitive edge in understanding and predicting stock market behavior.

## Future Work

- **Refinement of Sentiment Analysis**: Further tuning of transformer models and expanding the dataset can enhance the quality of the Greed and Fear index.
- **Broader Application**: Applying this sentiment-based approach to other financial instruments or indices to validate its utility across different markets.

## Acknowledgments

This project leverages datasets from [Kaggle](https://www.kaggle.com/aaron7sun/stocknews) and incorporates models from Hugging Face's extensive library of transformers.

