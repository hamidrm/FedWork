import socket
import threading
import struct
from utils.consts import *
from utils.common import *


class ClientComm:
    
    def __init__ (self, name, host, port, client_evt_fn) -> None:
        self.name = name
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.upload_data_size = 0
        self.upload_total_size = 0
        self.download_total_size = 0
        self.download_data_size = 0
        self.alive = True
        self.client_evt_fn = client_evt_fn
        self.recv_thread = threading.Thread(target=self.__recv_from_server)
        self.connect_to_server()
        self.recv_thread.start()
        
        

    def connect_to_server(self):
        self.socket.connect((self.host, self.port))
        self.client_evt_fn(self.name, COMM_EVT_CONNECTED, None)
        self.send_intro_to_server()

    def __recv_from_server(self):
        while self.alive:
            msg_header_bin = self.socket.recv(COMM_HEADER_SIZE) # Read the header
            if not msg_header_bin:
                continue
            header_data   = struct.unpack(COMM_HEADER_FORMAT, msg_header_bin)
            result_dict = dict(zip(COMM_HEADER_DICT.keys(), header_data))

            if result_dict["packet_sign"] == COMM_HEADER_SIGN:
                payload_size  = result_dict["payload_len"]
                packet_type   = result_dict["type"]
                packet_param1 = result_dict["param1"]
                packet_param2 = result_dict["param2"]

                self.download_total_size += payload_size + COMM_HEADER_SIZE
                if packet_type == COMM_HEADER_TYPES_CMD:
                    if packet_param1 == COMM_HEADER_CMD_NOP:
                        pass
                    elif packet_param1 == COMM_HEADER_CMD_TURNOFF:
                        self.alive = False
                        self.client_evt_fn(self.name, COMM_EVT_TURNOFF, None)
                    elif packet_param1 == COMM_HEADER_CMD_START_TRAINNING:
                        payload = None
                        if payload_size != 0:
                            payload = self.socket.recv(payload_size) # Read the payload
                            self.download_data_size += payload_size
                        self.client_evt_fn(self.name, COMM_EVT_TRAINING_START, Common.data_convert_from_bytes(payload))
                    else:
                        raise ValueError(f'Invalid command on client"{self.name}".')
                elif packet_type == COMM_HEADER_TYPES_DATA:
                    self.download_data_size += payload_size
                    payload = self.socket.recv(payload_size) # Read the payload
                    self.client_evt_fn(self.name, COMM_EVT_MODEL, Common.data_convert_from_bytes(payload))
                elif packet_type == COMM_HEADER_TYPES_NOTI:
                    pass
        self.client_evt_fn(self.name, COMM_EVT_DISCONNECTED, {"reason":"server"})

    def send_notification_to_server(self, notify_evt, param):
        header_data   = struct.pack(COMM_HEADER_FORMAT, COMM_HEADER_SIGN, 0, COMM_HEADER_TYPES_NOTI, notify_evt, param)
        self.upload_total_size += COMM_HEADER_SIZE
        self.socket.sendall(header_data)

    def send_data_to_server(self, data):
        payload_to_send = Common.data_convert_to_bytes(data)
        header_data   = struct.pack(COMM_HEADER_FORMAT, COMM_HEADER_SIGN, len(payload_to_send), COMM_HEADER_TYPES_DATA, 0, 0)
        self.upload_total_size += len(payload_to_send) + COMM_HEADER_SIZE
        self.upload_data_size += len(payload_to_send)
        self.socket.sendall(header_data)
        self.socket.sendall(payload_to_send)

    def disconnect(self):
        self.alive = False
        self.socket.close()
        self.client_evt_fn(self.name, COMM_EVT_DISCONNECTED, {"reason":"client"})

    def send_intro_to_server(self):
        info = {}
        info["name"] = self.name
        info["processing_power"] = 5 #Between 0 to 10
        payload_to_send = Common.data_convert_to_bytes(info)
        header_data   = struct.pack(COMM_HEADER_FORMAT, COMM_HEADER_SIGN, len(payload_to_send), COMM_HEADER_TYPES_INTRODUCTION, 0, 0)
        self.upload_total_size += len(payload_to_send) + COMM_HEADER_SIZE
        self.upload_data_size += len(payload_to_send)
        self.socket.sendall(header_data)
        self.socket.sendall(payload_to_send)
