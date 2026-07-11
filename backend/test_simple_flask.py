#!/usr/bin/env python
"""Simple Flask test"""
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/test', methods=['GET'])
def test_route():
    return jsonify({'status': 'ok', 'message': 'Simple Flask works!'}), 200

if __name__ == '__main__':
    print("Starting simple Flask server on 127.0.0.1:5000...")
    app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False, threaded=False)
