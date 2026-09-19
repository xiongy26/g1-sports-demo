#!/usr/bin/env python3
"""本地开发服务器：正确的 .wasm MIME + 禁用缓存。用法: python3 serve.py [端口]"""
import http.server
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


with Server(("0.0.0.0", PORT), Handler) as httpd:
    print(f"serving at http://localhost:{PORT}")
    httpd.serve_forever()
