# predictor.py
class Predictor:
    def __init__(self, model_handler):
        self.model_handler = model_handler

    def validate_inputs(self, context, question, model_path):
        if not context or not question:
            raise ValueError("Both 'context' and 'question' fields are required")
        if not model_path:
            raise ValueError("Model path is required")
            
    def run_inference(self, qa_pipeline, context, question):
        return qa_pipeline(question=question, context=context)

    def predict(self, context, question, model_path):
        try:
            self.validate_inputs(context, question, model_path)
            qa_pipeline = self.model_handler.get_model_pipeline(model_path)
            result = self.run_inference(qa_pipeline, context, question)

            return {
                "answer": result["answer"],
                "start": result["start"],
                "end": result["end"],
                "score": round(result["score"] * 100, 1),
            }

        except Exception as e:
            raise Exception(f"Error processing request: {str(e)}")