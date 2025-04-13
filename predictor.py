class Predictor:
    def __init__(self, model_handler):
        self.model_handler = model_handler
    
    def predict(self, context, question, model_path):
        if not context or not question:
            raise ValueError("Both 'context' and 'question' fields are required")
        
        if not model_path:
            raise ValueError("Model path is required")

        try:
            # Get the pipeline for the selected model
            qa_pipeline = self.model_handler.get_model_pipeline(model_path)
            
            # Run inference
            result = qa_pipeline(question=question, context=context, max_answer_len=100)
            
            return {
                "answer": result["answer"],
                "start": result["start"],
                "end": result["end"],
                "score": result["score"],
                "model": model_path
            }
        except Exception as e:
            raise Exception(f"Error processing request: {str(e)}")