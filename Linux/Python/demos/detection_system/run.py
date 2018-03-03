from yowsup.stacks import  YowStackBuilder
from .layer import EchoLayer
from yowsup.layers.auth import AuthError
from yowsup.layers import YowLayerEvent
from yowsup.layers.network import YowNetworkLayer

credentials = ("559884091755", "1BskXyPz15OjgrHdCyPfw4SHY4Q=")
if __name__==  "__main__":
        print("Inicializando")
        stackBuilder = YowStackBuilder()

        stack = stackBuilder\
            .pushDefaultLayers(True)\
            .push(EchoLayer)\
            .build()

        stack.setCredentials(credentials)

        stack.broadcastEvent(YowLayerEvent(YowNetworkLayer.EVENT_STATE_CONNECT))
        try:
            stack.loop()
        except AuthError as e:
            print("Authentication Error: %s" % e.message)
