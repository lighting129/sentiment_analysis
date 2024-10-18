import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer from the saved directory
model_name_or_path = "saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

# Define a function to perform sentiment analysis
def sentiment_analysis(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get the model's prediction
    outputs = model(**inputs)
    # Get the predicted label (0 or 1)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    
    # Map label to human-readable sentiment
    label_map = {0: "Negative", 1: "Positive"}
    sentiment = label_map.get(prediction, "Unknown")
    
    # Get the confidence score
    confidence = torch.softmax(outputs.logits, dim=1).max().item()

    return f"{sentiment} with a confidence score of {confidence:.2f}"

# Create the Gradio interface
interface = gr.Interface(
    fn=sentiment_analysis,
    inputs="text",
    outputs="text",
    title="Sentiment Analysis with BERT",
    description="Enter a movie review, and the model will predict whether it's positive or negative."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()