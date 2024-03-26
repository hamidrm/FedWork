
import pickle
import socket
import threading
from utils.consts import *
from utils.common import *
import struct
from utils.logger import *
import copy

class ServerComm:

    def __init__(self, host, port, server_evt_fn):
        self.host = host
        self.port = port
        self.server_ready = threading.Lock()
        self.server_ready.acquire()
        server_thread = threading.Thread(target=self.__server_thread)
        self.clients = {}
        self.alive = True
        self.server_evt_fn = server_evt_fn
        self.upload_data_size = 0
        self.upload_total_size = 0
        self.download_total_size = 0
        self.download_data_size = 0
        server_thread.start()
        self.server_ready.acquire()  #Block the server until server network gets ready.
        logger.log_debug(f"Initialization done.")

    def __server_thread(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            self.server_ready.release()
            s.listen()
            while self.alive:
                conn, addr = s.accept()
                self.__new_conection(conn, addr)

    def __new_conection(self, connection, address):
        
        try:
            msg_header_bin = connection.recv(COMM_HEADER_SIZE) # Read the header
            if not msg_header_bin:
                connection.close()
                return
        except socket.timeout:
            logger.log_warning(f"Client '{address}' timeout!")
            return None
        except ConnectionResetError:
            logger.log_error(f"Client '{address}' connection error!")
            return None
    
        logger.log_debug(f"New connection to '{address}'.")

        header_data   = struct.unpack(COMM_HEADER_FORMAT, msg_header_bin)
        result_dict = dict(zip(COMM_HEADER_DICT.keys(), header_data))
        self.download_total_size += len(msg_header_bin)
        if result_dict["packet_sign"] == COMM_HEADER_SIGN:
            packet_type   = result_dict["type"]
            #packet_param1 = result_dict["param1"]
            #packet_param2 = result_dict["param2"]
            if packet_type == COMM_HEADER_TYPES_INTRODUCTION:
                client_info = connection.recv(COMM_HCHUNK_TOTAL_DATA_SIZE) #Read the payload
                info = pickle.loads(client_info)
                client_data = ClientData(info["name"], address, info["processing_power"], connection)
                client_data.listener_thread = threading.Thread(target=self.__client_receiver, args=(client_data, ))
                self.clients[info["name"]] = client_data
                client_data.listener_thread.start()
                client_name = info["name"]
                
                logger.log_debug(f"Client '{client_name}' on address '{address}' has been successfully introduced and added to the clients' pool.")
            else:
                logger.log_error(f"Client '{client_name}' on address '{address}' introduction failed (introduction needed but wrong command received)!")
                connection.close()
        else:
            logger.log_error(f"Client '{client_name}' on address '{address}' introduction error (packet_sign)!")
            connection.close()

    def __client_receiver(self, client_data = None):
        if client_data == None:
            return
        client_is_alive = True
        self.server_evt_fn(COMM_EVT_CONNECTED, client_data, None)
        while(client_is_alive):
            packet_type, packet_param1, _, _, data = self.receive_data(SERVER_NAME, client_data.connection)
            if packet_type == COMM_HEADER_TYPES_CMD:
                logger.log_debug(f"Command received from '{client_data.name}' (packet_param1={packet_param1}).")
                if packet_param1 == COMM_HEADER_CMD_NOP:
                    pass
                elif packet_param1 == COMM_HEADER_CMD_TRAINNING_DONE:
                    self.server_evt_fn(COMM_EVT_TRAINING_DONE, client_data, None)
                elif packet_param1 == COMM_HEADER_CMD_DROPEME_REQ:
                    self.server_evt_fn(COMM_EVT_DROPEME_REQ, client_data, None)
                    client_is_alive = False             
                else:
                    logger.log_error(f'Invalid command on Server from "{client_data.name} ({client_data.addr})".')
                    continue
            elif packet_type == COMM_HEADER_TYPES_DATA:
                payload_dict = Common.data_convert_from_bytes(bytes(data))
                self.server_evt_fn(COMM_EVT_MODEL, client_data, payload_dict)
            elif packet_type == COMM_HEADER_TYPES_NOTI:
                logger.log_debug(f"Notification received from '{client_data.name}' (packet_param1={packet_param1}).")
                if packet_param1 == COMM_HEADER_NOTI_EPOCH_DONE:   
                    self.server_evt_fn(COMM_EVT_EPOCH_DONE_NOTIFY, client_data, Common.data_convert_from_bytes(data))
        client_data.connection.close()
        logger.log_debug(f"Connection closed...")
        del self.clients[client_data.name]
        self.server_evt_fn(COMM_EVT_DISCONNECTED, client_data, None)

    def get_clients(self):
        return self.clients
    
    def send_data(self, client_name, data):
        data_bytes_array = Common.data_convert_to_bytes(data)
        logger.log_debug(f"Sending data to '{client_name}'(size of data = {len(data_bytes_array)})")
        self.send_data(client_name, COMM_HEADER_TYPES_DATA, 0, 0, data_bytes_array)

    def send_command(self, client_name, command, param, data = None):
        logger.log_debug(f"Sending command to '{client_name}'(command={command}, size of data = {len(data)})")
        data_bytes_array = []
        if data is not None:
            data_bytes_array = Common.data_convert_to_bytes(data)
        self.send_data(client_name, COMM_HEADER_TYPES_CMD, command, param, data_bytes_array)

    def disconnect(self, client_name):
        self.clients[client_name].connection.close()
        logger.log_debug(f"Client '{client_name}' is disconnected...")
        self.server_evt_fn(COMM_EVT_DISCONNECTED, self.clients[client_name], None)


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
                data_buffer += data_to_send
                data_pad = bytes([0 for _ in range(COMM_HCHUNK_TOTAL_DATA_SIZE - data_length)])
                self.clients[receiver_id].connection.sendall(data_buffer+data_pad)
                if len(payload_to_send) <= COMM_HCHUNK_TOTAL_DATA_SIZE:
                    return
            else:
                data_length = min(len(payload_to_send) - data_offset, COMM_TCHUNK_TOTAL_DATA_SIZE)
                data_buffer = struct.pack(COMM_TAILS_FORMAT, COMM_TAILS_SIGN, receiver_id, data_length, data_seq_num)
                data_seq_num += 1
                data_to_send = payload_to_send[data_offset:data_offset+data_length]
                data_offset += data_length
                data_buffer += data_to_send
                data_pad = bytes([0 for _ in range(COMM_TCHUNK_TOTAL_DATA_SIZE - data_length)])
                self.clients[receiver_id].connection.sendall(data_buffer+data_pad)
                
                if data_offset == len(payload_to_send):
                    return

    def receive_data(self, recv_id, connection):
        expected_length = 0
        received_length = 0
        total_data = []
        seq_list = []

        while True:
            chunk = connection.recv(COMM_CHUNK_TOTAL_SIZE)

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

                if packet_seq == int((expected_length - COMM_HCHUNK_TOTAL_DATA_SIZE - 1) / COMM_TCHUNK_TOTAL_DATA_SIZE):
                    if received_length == expected_length:
                        return (packet_type, packet_param1, packet_param2, expected_length, bytes(total_data))
                    else:
                        expected_seq_list = [i for i in range(((expected_length - COMM_HCHUNK_TOTAL_DATA_SIZE) / COMM_TCHUNK_TOTAL_DATA_SIZE)) if i not in seq_list]
                        # We have lost some packets :(
                        # We must request resend them
                        for packet_num in expected_seq_list:
                            while True:
                                self.send_command(recv_id, COMM_HEADER_CMD_REQUEST_PACKET_NUM, packet_num, None)
                                chunk = connection.recv(COMM_CHUNK_TOTAL_SIZE)
                                
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