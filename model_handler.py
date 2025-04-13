import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

class ModelHandler:
    def __init__(self):
        # Dictionary to store loaded models
        self.loaded_models = {}
    
    def get_model_pipeline(self, model_path):
        if model_path in self.loaded_models:
            return self.loaded_models[model_path]
        
        try:
            model = AutoModelForQuestionAnswering.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
            
            # Store in cache
            self.loaded_models[model_path] = qa_pipeline
            return qa_pipeline
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            raise
    
    def get_available_models(self):
        models_directory = "models"
        models = []
        
        # Check if the models directory exists
        if os.path.exists(models_directory) and os.path.isdir(models_directory):
            # List all subdirectories in the models directory
            for model_folder in os.listdir(models_directory):
                model_path = os.path.join(models_directory, model_folder)
                
                # Check if it's a directory and contains model files
                if os.path.isdir(model_path) and self._is_valid_model(model_path):
                    # Extract the model name from configuration files if possible
                    model_name = self._get_model_name(model_path, model_folder)
                    
                    models.append({
                        "name": model_name,
                        "path": model_path
                    })
        
        return models
    
    def _is_valid_model(self, model_path):
        # Check for typical files that should exist in a Hugging Face model directory
        required_files = ["config.json"]
        
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                return False
        
        return True
    
    def _get_model_name(self, model_path, default_name):
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