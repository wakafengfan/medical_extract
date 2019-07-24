
# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request

from baseline.api.processor import ExtractProcessor

app = Flask(__name__)
p = ExtractProcessor()


@app.route('/medical_extractor', methods=['POST'])
def returnPost():
    print(request.get_json())
    text = request.json.get('text')
    tags = p.process(text)
    return jsonify({'tags': tags})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=21628)
