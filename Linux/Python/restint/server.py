###############################################################################
#
# The MIT License (MIT)
#
# Copyright (c) Crossbar.io Technologies GmbH
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
###############################################################################

import sys

from twisted.internet import reactor
from twisted.python import log

import json

from autobahn.twisted.websocket import WebSocketServerFactory, \
    WebSocketServerProtocol, \
    listenWS

#from libColetaFaceDB import MongoConn

class EchoServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        self.mdb = None

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        headers = {'MyCustomDynamicServerHeader1': 'Hello'}

        # Note: HTTP header field names are case-insensitive,
        # hence AutobahnPython will normalize header field names to
        # lower case.
        ##
        if 'mycustomclientheader' in request.headers:
            headers['MyCustomDynamicServerHeader2'] = request.headers['mycustomclientheader']
            print "onConnect: {}".format(headers['MyCustomDynamicServerHeader2'])
        # return a pair with WS protocol spoken (or None for any) and
        # custom headers to send in initial WS opening handshake HTTP response
        ##
        return (None, headers)

    def onMessage(self, payload, isBinary):
        self.sendMessage(payload, isBinary)
        #=======================================================================
        print "Payload: {}".format(payload)
        # raw = payload.decode('utf8')
        # msg = json.loads(raw)
        # print msg['type']
        # print "mongodb://{}:{}".format(msg['host'],msg['port'])
        # self.mdb = MongoConn(url="mongodb://{}:{}".format(msg['host'],msg['port']),
        #                         dbs=msg['base'])
        # resposta = {'RESPONSE':'OK'}
        # self.sendMessage(json.dumps(resposta))
        #=======================================================================


if __name__ == '__main__':

    log.startLogging(sys.stdout)

    headers = {'MyCustomServerHeader': 'Foobar'}

    factory = WebSocketServerFactory(u"ws://127.0.0.1:8082")
    factory.protocol = EchoServerProtocol
    listenWS(factory)

    reactor.run()
