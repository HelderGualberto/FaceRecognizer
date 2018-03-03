import websocket
#import asyncio
#import websockets

# kurento
#ws = websocket.WebSocket()
#ws.connect("ws://192.168.10.234:8888/kurento")

# servidor do helder
#s2 = websocket.WebSocket()
#ws2.connect("wss://192.168.10.234:8443/media", verify=False)
#ws2 = websocket.WebSocketApp("wss://192.168.10.234:8443/media")
#ws2.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
ws2 = websocket.create_connection("wss://192.168.10.234:8443/media", sslopt={"cert_reqs": ssl.CERT_NONE})
#ws2.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
