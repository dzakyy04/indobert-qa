# model_handler.py
import os
import json
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

class ModelHandler:
    def __init__(self):
        self.loaded_models = {}

    def load_model_and_tokenizer(self, model_path):
        try:
            model = AutoModelForQuestionAnswering.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return model, tokenizer
        except Exception as e:
            print(f"Error loading model or tokenizer from {model_path}: {str(e)}")
            raise

    def create_qa_pipeline(self, model, tokenizer):
        return pipeline("question-answering", model=model, tokenizer=tokenizer)

    def cache_pipeline(self, model_path, pipeline_instance):
        self.loaded_models[model_path] = pipeline_instance

    def get_model_pipeline(self, model_path):
        if model_path in self.loaded_models:
            cached_pipeline = self.loaded_models[model_path]
            return cached_pipeline

        try:
            model, tokenizer = self.load_model_and_tokenizer(model_path)
            qa_pipeline = self.create_qa_pipeline(model, tokenizer)
            self.cache_pipeline(model_path, qa_pipeline)
            return qa_pipeline
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            raise

    def get_available_models(self):
        models_directory = "models"
        models = []

        if os.path.exists(models_directory) and os.path.isdir(models_directory):
            for model_folder in os.listdir(models_directory):
                model_path = os.path.join(models_directory, model_folder)
                if os.path.isdir(model_path) and self._is_valid_model(model_path):
                    model_name = self._get_model_name(model_path, model_folder)
                    models.append({"name": model_name, "path": model_path})

        return models

    def _is_valid_model(self, model_path):
        required_files = ["config.json"]
        return all(os.path.exists(os.path.join(model_path, f)) for f in required_files)

    def _get_model_name(self, model_path, default_name):
        config_path = os.path.join(model_path, "config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    json.load(f)  # just to validate
                return default_name
        except:
            pass
        return default_name
