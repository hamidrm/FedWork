import socket
import threading
import struct
import copy
from utils.consts import *
from utils.common import *
from utils.logger import *
from network import Network

class ClientComm(Network):
    
    def __init__ (self, name, host, port, client_evt_fn) -> None:
        super().__init__()
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
        
        logger.log_debug(f"[{name}]: Initialization done.")
        
        

    def connect_to_server(self):
        self.socket.connect((self.host, self.port))
        self.client_evt_fn(COMM_EVT_CONNECTED, None)
        self.send_intro_to_server()
        super().create_new_receiver(self.name, self.socket)

    def __recv_from_server(self):
        logger.log_debug(f"[{self.name}]: Receiving thread has been started.")
        while self.alive:

            packet_type, packet_param1, packet_param2, payload_size, data, _ = self.receive_data()
            
            if packet_type == COMM_HEADER_TYPES_CMD:
                logger.log_debug(f"[{self.name}]: Received a command (param1={packet_param1}).")
                if packet_param1 == COMM_HEADER_CMD_NOP:
                    pass
                elif packet_param1 == COMM_HEADER_CMD_TURNOFF:
                    self.alive = False
                    self.client_evt_fn(COMM_EVT_TURNOFF, None)
                elif packet_param1 == COMM_HEADER_CMD_START_TRAINNING:
                    self.client_evt_fn(COMM_EVT_TRAINING_START, Common.data_convert_from_bytes(data))
                elif packet_param1 == COMM_HEADER_CMD_GET_TOTAL_EPOCHS:
                    self.client_evt_fn(COMM_EVT_EPOCHS_TOTAL_COUNT_REQ, None)
                elif packet_param1 == COMM_HEADER_CMD_GET_TRAINING_COUNT:
                    self.client_evt_fn(COMM_EVT_TRAINING_TOTAL_COUNT_REQ, None)
                elif packet_param1 == COMM_HEADER_CMD_START_PERIODIC_MODE:
                    self.client_evt_fn(COMM_EVT_START_PERIODIC_TRAINING, Common.data_convert_from_bytes(data))
                elif packet_param1 == COMM_HEADER_CMD_STOP_PERIODIC_MODE:
                    self.client_evt_fn(COMM_EVT_STOP_PERIODIC_TRAINING, None)
                elif packet_param1 == COMM_HEADER_CMD_REQUEST_PACKET_NUM:
                    super().resend_chunk_data(self.socket, self.name, packet_param2)
                else:
                    logger.log_error(f'Invalid command on client"{self.name}".')
            elif packet_type == COMM_HEADER_TYPES_DATA:
                logger.log_debug(f"[{self.name}]: Received data (payload_size={payload_size}).")
                self.client_evt_fn(COMM_EVT_MODEL, Common.data_convert_from_bytes(bytes(data)))
            elif packet_type == COMM_HEADER_TYPES_NOTI:
                logger.log_debug(f"[{self.name}]: Received notification (param1={packet_param1}).")
                pass
        

        self.client_evt_fn(COMM_EVT_DISCONNECTED, {"reason":"server"})
        logger.log_debug(f"[{self.name}]: Receiving thread has been finished.")

    def send_notification_to_server(self, notify_evt, param, data = None):
        data_bytes_array = []
        logger.log_debug(f"[{self.name}]: Sending notification to the server (notify_evt={notify_evt}).")
        if data is not None:
            data_bytes_array = Common.data_convert_to_bytes(data)
        self.send_data(self.socket, self.name, COMM_HEADER_TYPES_NOTI, notify_evt, param, data_bytes_array)
     

    def send_data_to_server(self, data):
        
        payload_to_send = Common.data_convert_to_bytes(data)
        logger.log_debug(f"[{self.name}]: Sending data to the server (payload_size={len(payload_to_send)}).")
        self.send_data(self.socket, self.name, COMM_HEADER_TYPES_DATA, 0, 0, payload_to_send)

    def disconnect(self):
        logger.log_debug(f"[{self.name}]: Disconnecting...")
        self.alive = False
        self.socket.close()
        self.client_evt_fn(COMM_EVT_DISCONNECTED, {"reason":"client"})

    def send_intro_to_server(self):
        info = {}
        logger.log_debug(f"[{self.name}]: Sending introduction to the server.")
        info["name"] = self.name
        info["processing_power"] = 5 #Between 0 to 10
        payload_to_send = Common.data_convert_to_bytes(info)
        self.send_data(self.socket, SERVER_NAME, COMM_HEADER_TYPES_INTRODUCTION, 0, 0, payload_to_send)

    def send_command_to_server(self, cmd, param):
        logger.log_debug(f"[{self.name}]: Sending command to the server.")
        self.send_data(self.socket, self.name, COMM_HEADER_TYPES_CMD, cmd, param, None)