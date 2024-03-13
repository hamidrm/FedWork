
import pickle
import socket
import threading
from utils.consts import *
from utils.common import *
import struct

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


    def __server_thread(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            self.server_ready.release()
            s.listen()
            while self.alive:
                conn, addr = s.accept()
                self.__new_conection(conn, addr)

    def __new_conection(self, connection, address):
        msg_header_bin = connection.recv(COMM_HEADER_SIZE) # Read the header
        if not msg_header_bin:
            client_data.connection.close()
            return
        header_data   = struct.unpack(COMM_HEADER_FORMAT, msg_header_bin)
        result_dict = dict(zip(COMM_HEADER_DICT.keys(), header_data))
        self.download_total_size += len(msg_header_bin)
        if result_dict["packet_sign"] == COMM_HEADER_SIGN:
            payload_size  = result_dict["payload_len"]
            packet_type   = result_dict["type"]
            #packet_param1 = result_dict["param1"]
            #packet_param2 = result_dict["param2"]
            if packet_type == COMM_HEADER_TYPES_INTRODUCTION:
                client_info = connection.recv(payload_size) #Read the payload
                self.download_total_size += payload_size
                self.download_data_size += payload_size
                info = pickle.loads(client_info)
                client_data = ClientData(info["name"], address, info["processing_power"], connection)

                client_data.listener_thread = threading.Thread(target=self.__client_receiver, args=(client_data, ))
                client_data.listener_thread.start()

                self.clients[info["name"]] = client_data

    def __client_receiver(self, client_data = None):
        if client_data == None:
            return
        client_is_alive = True
        self.server_evt_fn(COMM_EVT_CONNECTED, client_data, None)
        while(client_is_alive):
            msg_header_bin = client_data.connection.recv(COMM_HEADER_SIZE) # Read the header
            self.download_total_size += len(msg_header_bin)
            if not msg_header_bin:
                client_data.connection.close()
                self.server_evt_fn(COMM_EVT_DISCONNECTED, client_data, data)
                client_is_alive = False

            header_data   = struct.unpack(COMM_HEADER_FORMAT, msg_header_bin)
            result_dict = dict(zip(COMM_HEADER_DICT.keys(), header_data))

            if result_dict["packet_sign"] == COMM_HEADER_SIGN:
                payload_size  = result_dict["payload_len"]
                packet_type   = result_dict["type"]
                packet_param1 = result_dict["param1"]
                packet_param2 = result_dict["param2"]
                if packet_type == COMM_HEADER_TYPES_CMD:
                    if packet_param1 == COMM_HEADER_CMD_NOP:
                        pass
                    elif packet_param1 == COMM_HEADER_CMD_TRAINNING_DONE:
                        self.server_evt_fn(COMM_EVT_TRAINING_DONE, client_data, None)
                    elif packet_param1 == COMM_HEADER_CMD_DROPEME_REQ:
                        self.server_evt_fn(COMM_EVT_DROPEME_REQ, client_data, None)
                        client_is_alive = False             
                    else:
                        raise ValueError(f'Invalid command on Server from "{client_data.name} ({client_data.addr})".')
                elif packet_type == COMM_HEADER_TYPES_DATA:
                    self.download_total_size += payload_size
                    self.download_data_size += payload_size
                    recv_buffer_size = client_data.connection.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                    payload = []
                    while True:
                        chunk = client_data.connection.recv(min(recv_buffer_size, payload_size - len(payload)))
                        if not chunk:
                            break
                        payload += chunk

                    data = pickle.loads(bytes(payload))
                    self.server_evt_fn(COMM_EVT_MODEL, client_data, data)
                elif packet_type == COMM_HEADER_TYPES_NOTI:
                    if packet_param1 == COMM_HEADER_NOTI_EPOCH_DONE:
                        self.server_evt_fn(COMM_EVT_EPOCH_DONE_NOTIFY, client_data, {"accuracy": packet_param2})
        client_data.connection.close()
        del self.clients[client_data.name]
        self.server_evt_fn(COMM_EVT_DISCONNECTED, client_data, None)

    def get_clients(self):
        return self.clients
    
    def send_data(self, client_name, data):
        data_bytes_array = Common.data_convert_to_bytes(data)
        header_data   = struct.pack(COMM_HEADER_FORMAT, COMM_HEADER_SIGN, len(data_bytes_array), COMM_HEADER_TYPES_DATA, 0, 0)
        self.upload_total_size += len(header_data) + len(data_bytes_array)
        self.upload_data_size += len(data_bytes_array)
        self.clients[client_name].connection.sendall(header_data)
        self.clients[client_name].connection.sendall(data_bytes_array)

    def send_command(self, client_name, command, param, data = None):
        data_bytes_array = []
        if data is not None:
            data_bytes_array = Common.data_convert_to_bytes(data)
        header_data   = struct.pack(COMM_HEADER_FORMAT, COMM_HEADER_SIGN, len(data_bytes_array), COMM_HEADER_TYPES_CMD, command, param)
        self.upload_total_size += len(header_data)
        self.clients[client_name].connection.sendall(header_data)

        if data is not None:
            self.upload_total_size += len(data_bytes_array)
            self.upload_data_size += len(data_bytes_array)
            self.clients[client_name].connection.sendall(data_bytes_array)

    def disconnect(self, client_name):
        self.clients[client_name].connection.close()
        self.server_evt_fn(COMM_EVT_DISCONNECTED, self.clients[client_name], None)
