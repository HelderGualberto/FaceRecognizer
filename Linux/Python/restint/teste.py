
from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
    
import json

from twisted.python import log
from twisted.internet import reactor
import sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8080,
                    help='WebSocket Port')
args = parser.parse_args()


class OpenFaceServerProtocol(WebSocketServerProtocol):
    #def __init__(self):
    #    print "Init server"

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        self.training = True

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')

        #msg = json.loads(raw)
        print raw
        


if __name__ == '__main__':
    log.startLogging(sys.stdout)

    factory = WebSocketServerFactory("ws://localhost:{}".format(args.port))
    factory.protocol = OpenFaceServerProtocol

    reactor.listenTCP(args.port, factory)
    reactor.run()
    
    
    