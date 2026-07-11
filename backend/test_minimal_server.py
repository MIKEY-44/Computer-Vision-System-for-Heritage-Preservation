#!/usr/bin/env python
"""Test minimal Flask app"""
from flask import Flask, jsonify
from waitress import serve

app = Flask(__name__)

@app.route('/test')
def test():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("Starting minimal test server...")
    serve(app, host='127.0.0.1', port=5001, _quiet=True)
