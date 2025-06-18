from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from brain import predict_new_patient

app = Flask(__name__)
CORS(app)

REQUIRED_FIELDS = {
    "gender": str,
    "age": (int, float, str),
    "hypertension": (int, str),
    "heart_disease": (int, str),
    "ever_married": str,
    "work_type": str,
    "Residence_type": str,
    "avg_glucose_level": (float, int, str),
    "bmi": (float, int, str),
    "smoking_status": str
}

def validate_and_clean_data(data):
    cleaned = {}
    missing_fields = []

    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in data:
            missing_fields.append(field)
            continue
        
        value = data[field]

        if expected_type in [(int, float, str), (float, int, str)]:
            try:
                if field in ['age', 'hypertension', 'heart_disease']:
                    cleaned[field] = int(value)
                else:
                    cleaned[field] = float(value)
            except (ValueError, TypeError):
                raise ValueError(f"Field '{field}' must be a number.")
        else:
            if not isinstance(value, str):
                raise ValueError(f"Field '{field}' must be a string.")
            cleaned[field] = value

    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    return cleaned

@app.route('/', methods=['GET', 'POST'])
def stroke_prediction():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        try:
            cleaned_data = validate_and_clean_data(data)
            result = predict_new_patient(cleaned_data)
            return jsonify(result), 200
        except ValueError as ve:
            return jsonify({"error": f"Validation error: {str(ve)}"}), 400
        except Exception as e:
            return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    return jsonify({"error": "Method not allowed"}), 405

if __name__ == '__main__':
    app.run(debug=True)
