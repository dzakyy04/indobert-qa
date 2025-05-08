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
        return pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            max_answer_len=500,
            handle_impossible_answer=True
        )

    def cache_pipeline(self, model_path, pipeline_instance):
        self.loaded_models[model_path] = pipeline_instance

    def get_model_pipeline(self, model_path):
        if model_path in self.loaded_models:
            return self.loaded_models[model_path]
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
        folder_name = os.path.basename(model_path)
        config_path = os.path.join(model_path, "config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    json.load(f)
        except Exception:
            pass
        folder_parts = folder_name.split('_')
        model_num = ''.join(filter(str.isdigit, folder_parts[0]))
        lr_part = next((part for part in folder_parts if part.startswith('lr')), '')
        learning_rate = lr_part.replace('lr', '') if lr_part else ''
        bs_part = next((part for part in folder_parts if part.startswith('bs')), '')
        batch_size = bs_part.replace('bs', '') if bs_part else ''
        
        # Menentukan tipe penjawab berdasarkan kata kunci
        if 'selektif' in folder_parts:
            label = 'Penjawab Selektif'
        elif 'pasti' in folder_parts:
            label = 'Pasti Menjawab'
        else:
            label = 'Tidak Diketahui'
        
        # Menambahkan emoticon untuk model 3 dan 7 yang terbaik
        emoticon = ""
        if "terbaik" in folder_parts:
            if model_num == "3":
                emoticon = "🥇 "
            elif model_num == "7":
                emoticon = "🥇 "
        
        if model_num and learning_rate and batch_size:
            formatted_name = f"{emoticon}Model {model_num} ({label}) | LR: {learning_rate}, BS: {batch_size}"
            return formatted_name
        return default_name