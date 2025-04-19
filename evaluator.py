# evaluator.py
import pandas as pd
import evaluate
import uuid
import io

class Evaluator:
    def __init__(self, model_handler, cache):
        self.model_handler = model_handler
        self.cache = cache
        self.metric = evaluate.load("squad")

    def validate_file(self, file):
        if file.filename == '':
            raise ValueError('No file selected')
        if not file.filename.endswith('.csv'):
            raise ValueError('Uploaded file must be in CSV format')

    def generate_eval_id(self):
        return str(uuid.uuid4())

    def load_csv(self, file):
        df = pd.read_csv(file)
        required_columns = ["konteks", "pertanyaan", "jawaban"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f'CSV must contain columns: {required_columns}')

        df["id"] = df.index.astype(str)
        return df

    def calculate_predictions(self, df, qa_pipeline):
        df["prediksi"] = df.apply(
            lambda row: qa_pipeline(question=row["pertanyaan"], context=row["konteks"])["answer"],
            axis=1
        )
        return df

    def compute_metrics(self, df):
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

        overall_results = self.metric.compute(predictions=predictions, references=references)

        row_metrics = []
        for i, row in df.iterrows():
            row_pred = {"id": row["id"], "prediction_text": row["prediksi"]}
            row_ref = {"id": row["id"], "answers": {"text": [row["jawaban"]], "answer_start": [0]}}
            row_result = self.metric.compute(predictions=[row_pred], references=[row_ref])
            row_metrics.append({
                "exact_match": row_result["exact_match"],
                "f1": row_result["f1"]
            })

        df["exact_match"] = [m["exact_match"] for m in row_metrics]
        df["f1_score"] = [round(m["f1"], 2) for m in row_metrics]

        return df, overall_results
        
    def generate_download_link(self, eval_id):
        return f"/api/download-csv/{eval_id}"
    
    # New function to cache evaluation results
    def cache_evaluation_results(self, eval_id, df_export):
        self.cache.set(f'eval_{eval_id}', df_export.to_dict())

    def evaluate_model(self, file, model_path):
        self.validate_file(file)
        eval_id = self.generate_eval_id()

        try:
            df = self.load_csv(file)
            qa_pipeline = self.model_handler.get_model_pipeline(model_path)
            df = self.calculate_predictions(df, qa_pipeline)
            df, overall_metrics = self.compute_metrics(df)

            df_export = df[["konteks", "pertanyaan", "jawaban", "prediksi", "exact_match", "f1_score"]]
            # Use the new function to cache results
            self.cache_evaluation_results(eval_id, df_export)

            evaluation_data = df_export.to_dict(orient='records')
            download_link = self.generate_download_link(eval_id)

            results = {
                'eval_id': eval_id,
                'metrics': {
                    'exact_match': round(overall_metrics['exact_match'], 2),
                    'f1_score': round(overall_metrics['f1'], 2)
                },
                'evaluation_data': evaluation_data,
                'download_link': download_link
            }

            return results

        except Exception as e:
            raise Exception(str(e))

    def get_csv_export(self, eval_id):
        df_dict = self.cache.get(f'eval_{eval_id}')
        if df_dict is None:
            raise ValueError('Evaluation results not found')

        df_export = pd.DataFrame.from_dict(df_dict)
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()