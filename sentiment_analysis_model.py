import pickle

class SentimentAnalysisModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict_sentiment(self, sentence):
        prediction = self.model.predict([sentence])
        print("predication -------------->",prediction)
        return prediction[0]  # Assuming prediction is a single value

    def predict(self, tokenized_sentence):
        prediction = self.model.predict([tokenized_sentence])
        print("predication -------------->", prediction)
        return prediction[0]
