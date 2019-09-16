
# -*- coding: utf-8 -*-
import logging

from flask import Flask, jsonify, request

from baseline_3.api.processor import ExtractProcessor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
p = ExtractProcessor()


@app.route('/medical_extractor_v2', methods=['POST'])
def returnPost():
    logger.info(request.get_json())
    text = request.json.get('text')
    if len(text) > 500:
        return jsonify({'error': '句长需小于512'})

    tags = p.process(text)
    return jsonify({'tags': tags})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=21628)
