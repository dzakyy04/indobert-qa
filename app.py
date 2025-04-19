# app.py
from flask import Flask, render_template, request, jsonify, Response
from flask_caching import Cache
from model_handler import ModelHandler
from predictor import Predictor
from evaluator import Evaluator

class App:
    def __init__(self):
        self.app = Flask(__name__)

        cache_config = {
            "CACHE_TYPE": "SimpleCache",
            "CACHE_DEFAULT_TIMEOUT": 86400
        }
        self.cache = Cache(self.app, config=cache_config)

        self.model_handler = ModelHandler()
        self.predictor = Predictor(self.model_handler)
        self.evaluator = Evaluator(self.model_handler, self.cache)

        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/prediction')
        def prediction():
            return render_template('prediction.html')

        @self.app.route('/evaluation')
        def evaluation():
            return render_template('evaluation.html')

        @self.app.route('/api/get_available_models')
        def get_available_models():
            models = self.model_handler.get_available_models()
            return jsonify({"models": models})

        @self.app.route("/api/predict", methods=["POST"])
        def predict():
            data = request.json
            context = data.get("context", "").strip()
            question = data.get("question", "").strip()
            model_path = data.get("model_path", "models/model1").strip()

            try:
                result = self.predictor.predict(context, question, model_path)
                return jsonify(result)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/evaluate', methods=['POST'])
        def evaluate_model():
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file uploaded'}), 400

                file = request.files['file']
                model_path = request.form.get('model_path', 'models/model1')

                results = self.evaluator.evaluate_model(file, model_path)
                download_link = request.host_url.rstrip('/') + results['download_link']
                results['download_link'] = download_link

                return jsonify(results)

            except ValueError as e:
                return jsonify({'error': str(e)}), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/download-csv/<eval_id>', methods=['GET'])
        def download_csv(eval_id):
            try:
                csv_data = self.evaluator.get_csv_export(eval_id)
                response = Response(
                    csv_data,
                    mimetype='text/csv',
                    headers={
                        'Content-Disposition': f'attachment; filename=hasil_evaluasi.csv',
                        'Content-Type': 'text/csv'
                    }
                )
                return response
            except ValueError as e:
                return jsonify({'error': str(e)}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def run(self, debug=True, host='0.0.0.0', port=5000):
        self.app.run(debug=debug, host=host, port=port)

if __name__ == '__main__':
    app = App()
    app.run()
