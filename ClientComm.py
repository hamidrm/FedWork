import socket
import threading
import struct
from utils.consts import *
from utils.common import *
from utils.logger import *

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
        logger.log_debug(f"[{name}]: Initialization done.")
        
        

    def connect_to_server(self):
        self.socket.connect((self.host, self.port))
        self.client_evt_fn(COMM_EVT_CONNECTED, None)
        self.send_intro_to_server()

    def __recv_from_server(self):
        logger.log_debug(f"[{self.name}]: Receiving thread has been started.")
        while self.alive:
            try:
                msg_header_bin = self.socket.recv(COMM_HEADER_SIZE) # Read the header
                if not msg_header_bin:
                    continue
            except socket.timeout:
                logger.log_warning(f"[{self.name}]: Receiving timeout!")
                return None
            except ConnectionResetError:
                logger.log_error(f"[{self.name}]: Server disconnected unexpectedly!")
                break
            
            header_data   = struct.unpack(COMM_HEADER_FORMAT, msg_header_bin)
            result_dict = dict(zip(COMM_HEADER_DICT.keys(), header_data))

            if result_dict["packet_sign"] == COMM_HEADER_SIGN:
                payload_size  = result_dict["payload_len"]
                packet_type   = result_dict["type"]
                packet_param1 = result_dict["param1"]
                packet_param2 = result_dict["param2"]

                self.download_total_size += payload_size + COMM_HEADER_SIZE
                if packet_type == COMM_HEADER_TYPES_CMD:
                    logger.log_debug(f"[{self.name}]: Received a command (param1={packet_param1}).")
                    if packet_param1 == COMM_HEADER_CMD_NOP:
                        pass
                    elif packet_param1 == COMM_HEADER_CMD_TURNOFF:
                        self.alive = False
                        self.client_evt_fn(COMM_EVT_TURNOFF, None)
                    elif packet_param1 == COMM_HEADER_CMD_START_TRAINNING:
                        payload = None
                        if payload_size != 0:
                            try:
                                payload = self.socket.recv(payload_size) # Read the payload
                                if not payload:
                                    break
                            except socket.timeout:
                                logger.log_error(f"[{self.name}] Timeout in receiving start training data!")
                                continue
                            except ConnectionResetError:
                                logger.log_error(f"[{self.name}] disconnected unexpectedly during receiving start training data!")
                                break
                            self.download_data_size += payload_size
                        self.client_evt_fn(COMM_EVT_TRAINING_START, Common.data_convert_from_bytes(payload))
                    elif packet_param1 == COMM_HEADER_CMD_GET_TOTAL_EPOCHS:
                        self.client_evt_fn(COMM_EVT_EPOCHS_TOTAL_COUNT_REQ, None)
                    elif packet_param1 == COMM_HEADER_CMD_GET_TRAINING_COUNT:
                        self.client_evt_fn(COMM_EVT_TRAINING_TOTAL_COUNT_REQ, None)
                    elif packet_param1 == COMM_HEADER_CMD_START_PERIODIC_MODE:
                        payload = None
                        if payload_size != 0:
                            try:
                                payload = self.socket.recv(payload_size) # Read the payload
                                if not payload:
                                    break
                            except socket.timeout:
                                logger.log_error(f"[{self.name}] Timeout in receiving start training data!")
                                continue
                            except ConnectionResetError:
                                logger.log_error(f"[{self.name}] disconnected unexpectedly during receiving start training data!")
                                break
                            self.download_data_size += payload_size
                        self.client_evt_fn(COMM_EVT_START_PERIODIC_TRAINING, Common.data_convert_from_bytes(payload))
                    elif packet_param1 == COMM_HEADER_CMD_STOP_PERIODIC_MODE:
                        self.client_evt_fn(COMM_EVT_STOP_PERIODIC_TRAINING, None)
                    else:
                        logger.log_error(f'Invalid command on client"{self.name}".')
                elif packet_type == COMM_HEADER_TYPES_DATA:
                    logger.log_debug(f"[{self.name}]: Received data (payload_size={payload_size}).")
                    self.download_data_size += payload_size

                    recv_buffer_size = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                    payload = []
                    while True:
                        try:
                            chunk = self.socket.recv(min(recv_buffer_size, payload_size - len(payload)))
                            if not chunk:
                                break
                        except socket.timeout:
                            logger.log_error(f"[{self.name}] Timeout in model receiving!")
                            continue
                        except ConnectionResetError:
                            logger.log_error(f"[{self.name}] disconnected unexpectedly during model receiving!")
                            break

                        payload += chunk

                    self.client_evt_fn(COMM_EVT_MODEL, Common.data_convert_from_bytes(bytes(payload)))
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
        header_data   = struct.pack(COMM_HEADER_FORMAT, COMM_HEADER_SIGN, len(data_bytes_array), COMM_HEADER_TYPES_NOTI, notify_evt, param)
        self.upload_total_size += COMM_HEADER_SIZE
        self.socket.sendall(header_data)

        if data is not None:
            self.upload_total_size += len(data_bytes_array)
            self.upload_data_size += len(data_bytes_array)
            self.socket.sendall(data_bytes_array)

    def send_data_to_server(self, data):
        
        payload_to_send = Common.data_convert_to_bytes(data)
        header_data   = struct.pack(COMM_HEADER_FORMAT, COMM_HEADER_SIGN, len(payload_to_send), COMM_HEADER_TYPES_DATA, 0, 0)
        self.upload_total_size += len(payload_to_send) + COMM_HEADER_SIZE
        self.upload_data_size += len(payload_to_send)
        logger.log_debug(f"[{self.name}]: Sending data to the server (payload_size={len(payload_to_send)}).")
        self.socket.sendall(header_data)
        self.socket.sendall(payload_to_send)

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
        header_data   = struct.pack(COMM_HEADER_FORMAT, COMM_HEADER_SIGN, len(payload_to_send), COMM_HEADER_TYPES_INTRODUCTION, 0, 0)
        self.upload_total_size += len(payload_to_send) + COMM_HEADER_SIZE
        self.upload_data_size += len(payload_to_send)
        self.socket.sendall(header_data)
        self.socket.sendall(payload_to_send)
