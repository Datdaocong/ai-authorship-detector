from predict import predict_text

sample_text = "Technology is rapidly changing the modern workplace."

prediction, confidence_scores = predict_text(sample_text)

print("Prediction:", prediction)
print("Confidence scores:", confidence_scores)