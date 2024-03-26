import socket
import threading
import struct
import copy
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
        self.send_buffer = []
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

            packet_type, packet_param1, _, payload_size, data = self.receive_data(self.name)
            
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
        self.send_data(SERVER_NAME, COMM_HEADER_TYPES_NOTI, notify_evt, param, data_bytes_array)
     

    def send_data_to_server(self, data):
        
        payload_to_send = Common.data_convert_to_bytes(data)
        logger.log_debug(f"[{self.name}]: Sending data to the server (payload_size={len(payload_to_send)}).")
        self.send_data(SERVER_NAME, COMM_HEADER_TYPES_DATA, 0, 0, payload_to_send)

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
        self.send_data(SERVER_NAME, COMM_HEADER_TYPES_INTRODUCTION, 0, 0, payload_to_send)

    def send_command_to_server(self, cmd, param):
        logger.log_debug(f"[{self.name}]: Sending command to the server.")
        self.send_data(SERVER_NAME, COMM_HEADER_TYPES_CMD, cmd, param, None)

    def send_data(self, receiver_id, type, param1, param2, data):
        data_offset = 0
        data_seq_num = 0
        
        payload_to_send = Common.data_convert_to_bytes(data)
        self.send_buffer = copy.deepcopy(payload_to_send)

        while True:

            if data_offset == 0:
                data_buffer = struct.pack(COMM_HEADER_FORMAT, COMM_HEADER_SIGN, len(payload_to_send), type, param1, param2, receiver_id.encode('ascii'))
                data_length = min(len(payload_to_send), COMM_HCHUNK_TOTAL_DATA_SIZE)
                data_to_send = payload_to_send[:data_length]
                data_offset += data_length
                data_pad = bytes([0 for _ in range(COMM_HCHUNK_TOTAL_DATA_SIZE - data_length)])
                self.socket.sendall(data_buffer+data_to_send+data_pad)
                if len(payload_to_send) <= COMM_HCHUNK_TOTAL_DATA_SIZE:
                    return
            else:
                data_length = min(len(payload_to_send) - data_offset, COMM_TCHUNK_TOTAL_DATA_SIZE)
                data_buffer = struct.pack(COMM_TAILS_FORMAT, COMM_TAILS_SIGN, receiver_id.encode('ascii'), data_length, data_seq_num)
                data_seq_num += 1
                data_to_send = payload_to_send[data_offset:data_offset+data_length]
                data_offset += data_length
                data_buffer += data_to_send
                data_pad = bytes([0 for _ in range(COMM_TCHUNK_TOTAL_DATA_SIZE - data_length)])
                self.socket.sendall(data_buffer+data_pad)
                
                if data_offset == len(payload_to_send):
                    return

    def receive_data(self, recv_id):
        expected_length = 0
        received_length = 0
        total_data = []
        seq_list = []

        while True:
            chunk = self.socket.recv(COMM_CHUNK_TOTAL_SIZE)

            header_data   = struct.unpack(COMM_HEADER_FORMAT, chunk[:COMM_HEADER_SIZE])
            header_dict = dict(zip(COMM_HEADER_DICT.keys(), header_data)) 

            if header_dict["packet_sign"] == COMM_HEADER_SIGN:
                payload_size  = header_dict["payload_len"]
                packet_type   = header_dict["type"]
                packet_param1 = header_dict["param1"]
                packet_param2 = header_dict["param2"]
                packet_id     = header_dict["id"].decode('ascii')

                if packet_id != recv_id:
                    continue

                expected_length = payload_size
                
                received_length = min(payload_size, COMM_HCHUNK_TOTAL_DATA_SIZE)

                total_data = [0] * expected_length

                total_data[:payload_size] = chunk[COMM_HEADER_SIZE:COMM_HEADER_SIZE+payload_size]

                if expected_length <= COMM_HCHUNK_TOTAL_DATA_SIZE:
                    return (packet_type, packet_param1, packet_param2, expected_length, bytes(total_data))
                
            elif header_dict["packet_sign"] == COMM_TAILS_SIGN:
                tails_data   = struct.unpack(COMM_TAILS_FORMAT, chunk[:COMM_TAILS_SIZE])
                tails_dict = dict(zip(COMM_TAILS_DICT.keys(), tails_data))

                payload_size  = tails_dict["payload_len"]
                packet_seq   = tails_dict["sequence"]
                packet_id     = tails_dict["id"].decode('ascii')

                if packet_id != recv_id:
                    continue
                
                seq_list.append(packet_seq)

                total_data[packet_seq * COMM_TCHUNK_TOTAL_DATA_SIZE + COMM_HCHUNK_TOTAL_DATA_SIZE:packet_seq * COMM_TCHUNK_TOTAL_DATA_SIZE + COMM_HCHUNK_TOTAL_DATA_SIZE + payload_size] = chunk[COMM_TAILS_SIZE:COMM_TAILS_SIZE+payload_size]

                received_length += payload_size

                if packet_seq == ((expected_length - COMM_HCHUNK_TOTAL_DATA_SIZE - 1) / COMM_TCHUNK_TOTAL_DATA_SIZE):
                    if received_length == expected_length:
                        return (packet_type, packet_param1, packet_param2, expected_length, bytes(total_data))
                    else:
                        expected_seq_list = [i for i in range(((expected_length - COMM_HCHUNK_TOTAL_DATA_SIZE) / COMM_TCHUNK_TOTAL_DATA_SIZE)) if i not in seq_list]
                        # We have lost some packets :(
                        # We must request resend them
                        for packet_num in expected_seq_list:
                            while True:
                                self.send_command_to_server(COMM_HEADER_CMD_REQUEST_PACKET_NUM, packet_num)
                                chunk = self.socket.recv(COMM_CHUNK_TOTAL_SIZE)
                                
                                tails_data   = struct.unpack(COMM_TAILS_FORMAT, chunk[:COMM_TAILS_SIZE])
                                tails_dict = dict(zip(COMM_TAILS_DICT.keys(), tails_data))

                                payload_size  = tails_dict["payload_len"]
                                packet_seq   = tails_dict["sequence"]
                                packet_id     = tails_dict["id"].decode('ascii')

                                if packet_id != recv_id:
                                    continue

                                if packet_seq != packet_num:
                                    continue

                                total_data[packet_seq * COMM_TCHUNK_TOTAL_DATA_SIZE + COMM_HCHUNK_TOTAL_DATA_SIZE:packet_seq * COMM_TCHUNK_TOTAL_DATA_SIZE + COMM_HCHUNK_TOTAL_DATA_SIZE + payload_size] = chunk[COMM_TAILS_SIZE:COMM_TAILS_SIZE+payload_size]
                                received_length += payload_size
                        return (packet_type, packet_param1, packet_param2, expected_length, bytes(total_data))