"""Local response-only dashboard. No external services or inference dependencies."""
import argparse
import json
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .common import LABELS, PRIORITY, Store, read


def handler(store, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Do not write responses or private annotation notes to logs.

        def send(self, status, data, mime='application/json'):
            raw=data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type',mime+'; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'nonce-"+token+"'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers();self.wfile.write(raw)

        def valid_host(self):
            return self.headers.get('Host') in ('127.0.0.1:'+str(self.server.server_port),'localhost:'+str(self.server.server_port))

        def do_GET(self):
            if not self.valid_host():return self.send(403,{'error':'Local host only'})
            if self.path=='/':
                page=Path(__file__).with_name('dashboard.html').read_text()
                page=page.replace('__TOKEN__',token)
                return self.send(200,page.encode(),'text/html')
            if self.path=='/state':
                return self.send(200,{'mode':store.bundle['mode'],'rows':store.bundle['rows'],
                                     'ratings':store.records(),'sealed':store.sealed(),'labels':LABELS,'priority':PRIORITY})
            if self.path=='/export':return self.send(200,store.export())
            self.send(404,{'error':'Not found'})

        def do_POST(self):
            origin=self.headers.get('Origin')
            if (not self.valid_host() or self.headers.get('X-Audit-Token')!=token
                    or (origin and origin!='http://'+self.headers.get('Host',''))):
                return self.send(403,{'error':'Invalid local session'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if size<0 or size>10000:raise ValueError('Request too large')
                data=json.loads(self.rfile.read(size))
                if self.path=='/save':store.save(data['audit_id'],data['rating'])
                elif self.path=='/seal':store.seal()
                else:return self.send(404,{'error':'Not found'})
                self.send(200,{'ok':True})
            except (ValueError,KeyError,TypeError) as e:self.send(400,{'error':str(e)})
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True);p.add_argument('--database',required=True);p.add_argument('--coder',required=True)
    p.add_argument('--port',type=int,default=8765);args=p.parse_args()
    store=Store(args.database,read(args.bundle),args.coder)
    server=HTTPServer(('127.0.0.1',args.port),handler(store,secrets.token_hex(24)))
    print('Open http://127.0.0.1:'+str(server.server_port),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();store.db.close()

if __name__=='__main__':main()
