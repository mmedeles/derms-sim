"""
Tiny MQTT client wrapper (paho-mqtt)
"""
import json, logging
import paho.mqtt.client as mqtt

log = logging.getLogger(__name__)

class MQTTClient:
    def __init__(self, client_id="derms", host="localhost", port=1883, keepalive=60):
        self.client = mqtt.Client(client_id=client_id)
        self.client.on_connect = lambda c,u,f,rc: log.info("MQTT connected rc=%s", rc)
        self.client.connect(host, port, keepalive)
        self.client.loop_start()

    def publish(self, topic, payload, qos=0, retain=False):
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload)
        self.client.publish(topic, payload, qos=qos, retain=retain)

    def subscribe(self, topic, on_message=None, qos=0):
        if on_message is not None:
            def _cb(client, userdata, msg):
                try:
                    on_message(client, userdata, msg)
                except Exception:  # keep loop alive
                    log.exception("on_message failed")
            self.client.on_message = _cb
        self.client.subscribe(topic, qos=qos)
