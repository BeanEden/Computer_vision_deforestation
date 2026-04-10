import os
import pandas as pd
from flask import Flask, render_template, jsonify, send_from_directory, request

app = Flask(__name__)

# Dossier des résultats
BASE_OUTPUTS_DIR = os.path.join(os.getcwd(), 'outputs')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data/<lot_id>')
def get_data(lot_id):
    lot_dir = os.path.join(BASE_OUTPUTS_DIR, lot_id)
    csv_path = os.path.join(lot_dir, 'bilan_final_expert.csv')
    
    if not os.path.exists(csv_path):
        return jsonify([])
    
    df = pd.read_csv(csv_path)
    return jsonify(df.to_dict(orient='records'))

@app.route('/outputs/<lot_id>/<path:filename>')
def serve_output(lot_id, filename):
    lot_dir = os.path.join(BASE_OUTPUTS_DIR, lot_id)
    return send_from_directory(lot_dir, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
