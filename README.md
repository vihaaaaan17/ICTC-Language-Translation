# Lingo Blend

This project focuses on text classification and translation tasks using a dataset containing text samples in six different languages written using English alphabets. The primary objectives are:

1. **Text Classification**: Classify the text samples into their respective languages.
2. **Text Translation**: Translate the classified text samples into English.

## Dataset

Dataset consists of text samples written in six different languages using English alphabets. Each sample is labeled with its corresponding language. The languages included in the dataset are [list the languages here].



![data](https://github.com/Vinayakgoyal24/ICTC2.0/assets/141870146/ae2dd2f3-7726-4661-9039-8d25f00a71c2)

## Approach

### Text Classification

We experimented with various classification models to determine the language of the text samples. The models we tried include:

- Naive Bayes
- Support Vector Machines (SVM)
- Logistic Regression
- Latent Dirichlet Allocation (LDA)
- Long Short-Term Memory (LSTM)

After evaluating the performance of each model, we found that an ensemble of Logistic Regression, LSTM, and Naive Bayes yielded the best results.

### Text Translation

For the translation task, we fine-tuned the Varta-T-5 model, a transformer-based model specialized in translation tasks. This model was chosen for its effectiveness in handling multilingual translation tasks.

## Implementation

### Text Classification

We used Python along with popular machine learning libraries such as scikit-learn and TensorFlow/Keras for implementing the classification models. The ensemble approach was implemented by combining the predictions from each individual model.


![classification](https://github.com/Vinayakgoyal24/ICTC2.0/assets/141870146/b9fbfef8-ea12-454b-a942-ee0549bd66f5)


### Text Translation

We employed the Hugging Face Transformers library for fine-tuning the Varta-T-5 model for text translation. The fine-tuning process involved training the model on the given dataset to optimize its performance for translating text samples into English.


![Blank diagram](https://github.com/Vinayakgoyal24/ICTC2.0/assets/141870146/8a7383ed-d484-42db-8a34-97b669f6effe)



## Usage

1. **Data Preprocessing**: Preprocess the dataset by cleaning and tokenizing the text samples.
2. **Text Classification**:
   - Train the individual classification models (Naive Bayes, Logistic Regression, LSTM).
   - Combine the predictions using ensemble techniques.
3. **Text Translation**:
   - Fine-tune the Varta-T-5 model on the provided dataset.
4. **Inference**:
   - Classify the text samples using the ensemble classifier.
   - Translate the classified text samples into English using the fine-tuned Varta-T-5 model.

## Results

The performance of the text classification and translation tasks was evaluated using metrics such as accuracy, precision, recall, and F1-score.
Results of classification are as follows:


![c result](https://github.com/Vinayakgoyal24/ICTC2.0/assets/141870146/2abc6bf0-135b-4979-9419-5b96920eaeb3)


## Dependencies

- Python 3.x
- scikit-learn
- TensorFlow
- Keras
- Hugging Face Transformers

## Conclusion

Our approach using an ensemble of logistic regression, LSTM, and Naive Bayes for text classification, along with fine-tuning the Varta-T-5 model for text translation, achieved satisfactory results. Further optimization and experimentation could potentially enhance the performance of the system.


