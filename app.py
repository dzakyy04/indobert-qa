from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import os

app = Flask(__name__)

# Dictionary to store loaded models
loaded_models = {}

def get_model_pipeline(model_path):
    if model_path in loaded_models:
        return loaded_models[model_path]
    
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
        
        # Store in cache
        loaded_models[model_path] = qa_pipeline
        return qa_pipeline
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/get_available_models')
def get_available_models():
    models_directory = "models"
    models = []
    
    # Check if the models directory exists
    if os.path.exists(models_directory) and os.path.isdir(models_directory):
        # List all subdirectories in the models directory
        for model_folder in os.listdir(models_directory):
            model_path = os.path.join(models_directory, model_folder)
            
            # Check if it's a directory and contains model files
            if os.path.isdir(model_path) and is_valid_model(model_path):
                # Extract the model name from configuration files if possible
                model_name = get_model_name(model_path, model_folder)
                
                models.append({
                    "name": model_name,
                    "path": model_path
                })
    
    return jsonify({"models": models})

def is_valid_model(model_path):
    # Check for typical files that should exist in a Hugging Face model directory
    required_files = ["config.json"]
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            return False
    
    return True

def get_model_name(model_path, default_name):
    # Try to read from config.json
    config_path = os.path.join(model_path, "config.json")
    try:
        import json
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return f"{default_name}"
    except:
        pass
    
    # If we couldn't get a better name, just use the directory name
    return default_name

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    context = data.get("context", "").strip()
    question = data.get("question", "").strip()
    model_path = data.get("model_path", "models/model1").strip()

    if not context or not question:
        return jsonify({"error": "Both 'context' and 'question' fields are required"}), 400
    
    if not model_path:
        return jsonify({"error": "Model path is required"}), 400

    try:
        # Get the pipeline for the selected model
        qa_pipeline = get_model_pipeline(model_path)
        
        # Run inference
        result = qa_pipeline(question=question, context=context, max_answer_len=100)
        
        return jsonify({
            "answer": result["answer"],
            "start": result["start"],
            "end": result["end"],
            "score": result["score"],
            "model": model_path
        })
    except Exception as e:
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
