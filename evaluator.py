import pandas as pd
import evaluate
import uuid
import io

class Evaluator:
    def __init__(self, model_handler, cache):
        self.model_handler = model_handler
        self.cache = cache
        self.metric = evaluate.load("squad")
    
    def evaluate_model(self, file, model_path):
        if file.filename == '':
            raise ValueError('No file selected')
        
        if not file.filename.endswith('.csv'):
            raise ValueError('Uploaded file must be in CSV format')
        
        # Generate a unique ID for this evaluation
        eval_id = str(uuid.uuid4())
        
        try:
            # Load the CSV file directly from the request
            df = pd.read_csv(file)
            
            # Check for required columns
            required_columns = ["konteks", "pertanyaan", "jawaban"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f'CSV must contain columns: {required_columns}')
            
            # Add ID column for internal use only
            df["id"] = df.index.astype(str)
            
            # Get the model pipeline
            qa_pipeline = self.model_handler.get_model_pipeline(model_path)
            
            # Calculate predictions
            df["prediksi"] = df.apply(
                lambda row: qa_pipeline(
                    question=row["pertanyaan"], 
                    context=row["konteks"]
                )["answer"], 
                axis=1
            )
            
            # Prepare data for evaluation
            references = [
                {
                    "id": row["id"],
                    "answers": {
                        "text": [row["jawaban"]],
                        "answer_start": [0]
                    }
                } for _, row in df.iterrows()
            ]
            
            predictions = [
                {
                    "id": row["id"],
                    "prediction_text": row["prediksi"]
                } for _, row in df.iterrows()
            ]
            
            # Calculate overall metrics
            results = self.metric.compute(predictions=predictions, references=references)
            
            # Calculate metrics for each row
            row_metrics = []
            for i, row in df.iterrows():
                row_pred = {"id": row["id"], "prediction_text": row["prediksi"]}
                row_ref = {"id": row["id"], "answers": {"text": [row["jawaban"]], "answer_start": [0]}}
                
                # Calculate metrics for this individual row
                row_result = self.metric.compute(
                    predictions=[row_pred],
                    references=[row_ref]
                )
                
                row_metrics.append({
                    "exact_match": row_result["exact_match"],
                    "f1": row_result["f1"]
                })
            
            # Add metrics to DataFrame
            df["exact_match"] = [m["exact_match"] for m in row_metrics]
            df["f1_score"] = [round(m["f1"], 2) for m in row_metrics]
            
            # Prepare results to send back in the API response
            evaluation_data = df[["konteks", "pertanyaan", "jawaban", "prediksi", "exact_match", "f1_score"]].to_dict(orient='records')
            
            # Create a filtered DataFrame for CSV export
            df_export = df[["konteks", "pertanyaan", "jawaban", "prediksi", "exact_match", "f1_score"]]
            
            # Store DataFrame in cache for later access
            self.cache.set(f'eval_{eval_id}', df_export.to_dict())
            
            return {
                'eval_id': eval_id,
                'model_path': model_path,
                'metrics': {
                    'exact_match': round(results['exact_match'], 2),
                    'f1_score': round(results['f1'], 2)
                },
                'evaluation_data': evaluation_data,
                'df_export': df_export
            }
        
        except Exception as e:
            raise Exception(str(e))
    
    def get_csv_export(self, eval_id):
        # Retrieve DataFrame from cache
        df_dict = self.cache.get(f'eval_{eval_id}')
        
        if df_dict is None:
            raise ValueError('Evaluation results not found')
        
        # Convert dict back to DataFrame
        df_export = pd.DataFrame.from_dict(df_dict)
        
        # Generate CSV in memory
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        return csv_data