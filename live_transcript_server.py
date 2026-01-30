#!/usr/bin/env python3
"""A tiny Flask app that streams words from a story file over Server-Sent Events (SSE).

Usage:
  pip install flask
  python live_transcript_server.py --file story5.txt --delay 1.0

Then open http://127.0.0.1:5000/ in your browser.

The server supports query params on /stream: file and delay, e.g. /stream?file=story5.txt&delay=0.5
Each SSE `message` contains a JSON object with fields: {"word": str, "is_start": bool}
"""
from __future__ import annotations
import argparse
import json
import os
import time
from urllib.parse import unquote
from flask import Flask, Response, render_template_string, request, send_from_directory

from live_transcript import line_word_generator

import logging

app = Flask(__name__, static_folder="web", template_folder="web")
# log to console at INFO so we can see requests in the terminal
app.logger.setLevel(logging.INFO)

INDEX_HTML = "index.html"


@app.route("/")
def index():
    # render the static index.html but substitute default values for file and delay
    try:
        with open(os.path.join(app.static_folder, INDEX_HTML), 'r', encoding='utf-8') as f:
            tpl = f.read()
    except Exception:
        app.logger.exception('Could not read index.html')
        return send_from_directory(app.static_folder, INDEX_HTML)

    default_file = app.config.get('DEFAULT_FILE', 'story5.txt')
    default_delay = app.config.get('DEFAULT_DELAY', 1.0)
    rendered = render_template_string(tpl, DEFAULT_FILE=default_file, DEFAULT_DELAY=default_delay)
    return rendered


@app.route("/stream")
def stream():
    # file path and delay can be passed as query params
    file = request.args.get("file", "story5.txt")
    delay = float(request.args.get("delay", "1.0"))

    # prevent directory traversal; only allow files from current dir
    file = os.path.basename(file)
    path = os.path.join(os.getcwd(), file)

    if not os.path.isfile(path):
        app.logger.info(f"/stream requested but file not found: {path}")
        return Response(f"File not found: {file}\n", status=404)

    app.logger.info(f"/stream requested: path={path}, delay={delay}")

    def event_stream():
        app.logger.info("event_stream starting")
        for w, is_start in line_word_generator(path):
            payload = json.dumps({"word": w, "is_start": is_start})
            yield f"data: {payload}\n\n"
            time.sleep(delay)
        app.logger.info("event_stream finished")
        # signal end
        yield "data: {\"event\":\"end\"}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


@app.route('/debug')
def debug():
    """Return debug info about the static folder and files."""
    try:
        files = os.listdir(app.static_folder)
    except Exception as e:
        files = [f"error: {e}"]
    found = 'main.js' in files
    app.logger.info(f"/debug called, static_folder={app.static_folder}, contains_main_js={found}")
    return Response(json.dumps({"static_folder": app.static_folder, "files": files, "main_js": found}), mimetype='application/json')


@app.before_request
def log_request_info():
    app.logger.info(f"request: method={request.method}, path={request.path}, args={request.args}")


@app.route('/static/main.js')
def serve_main_js():
    """Serve main.js explicitly (helps diagnose 404s)."""
    app.logger.info("/static/main.js requested; serving from web/main.js")
    return send_from_directory(app.static_folder, 'main.js')


@app.route('/main.js')
def serve_main_js_root():
    """Also serve from /main.js for compatibility (some browsers/tools request it)."""
    app.logger.info("/main.js requested; serving from web/main.js")
    return send_from_directory(app.static_folder, 'main.js')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="story5.txt", help="Story file to stream (must exist in current directory)")
    p.add_argument("--delay", type=float, default=0.3, help="Delay between words in seconds")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()

    # verify file exists
    if not os.path.isfile(args.file):
        print(f"Error: file not found: {args.file}")
        raise SystemExit(2)

    # make defaults available to index rendering
    app.config['DEFAULT_FILE'] = args.file
    app.config['DEFAULT_DELAY'] = args.delay

    print(f"Starting server on http://{args.host}:{args.port}/ - streaming {args.file} with {args.delay}s delay")
    # we tell users to pass file/delay via query params to /stream, but this makes a default
    app.run(host=args.host, port=args.port, threaded=True)
