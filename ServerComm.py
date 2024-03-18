
import pickle
import socket
import threading
from utils.consts import *
from utils.common import *
import struct
from utils.logger import *

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
                client_name = info["name"]
                self.clients[info["name"]] = client_data
                logger.log_debug(f"Client '{client_name}' on address '{address}' has been introduced successfully.")
            else:
                logger.log_error(f"Client '{client_name}' on address '{address}' introduction error (introduction needed)!")
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
            try:
                msg_header_bin = client_data.connection.recv(COMM_HEADER_SIZE) # Read the header
                if not msg_header_bin:
                    client_data.connection.close()
                    self.server_evt_fn(COMM_EVT_DISCONNECTED, client_data, data)
                    client_is_alive = False
            except socket.timeout:
                logger.log_error(f"Client '{client_data.name}' timeout!")
                client_is_alive = False
                break
            except ConnectionResetError:
                logger.log_error(f"Client '{client_data.name}' disconnected unexpectedly!")
                client_is_alive = False
                break
    

            
            self.download_total_size += len(msg_header_bin)

            header_data   = struct.unpack(COMM_HEADER_FORMAT, msg_header_bin)
            result_dict = dict(zip(COMM_HEADER_DICT.keys(), header_data))

            if result_dict["packet_sign"] == COMM_HEADER_SIGN:
                payload_size  = result_dict["payload_len"]
                packet_type   = result_dict["type"]
                packet_param1 = result_dict["param1"]
                packet_param2 = result_dict["param2"]
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
                    self.download_total_size += payload_size
                    self.download_data_size += payload_size
                    logger.log_debug(f"Data received from '{client_data.name}' (payload_size={payload_size}).")
                    recv_buffer_size = client_data.connection.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                    payload = []
                    while True:
                        try:
                            chunk = client_data.connection.recv(min(recv_buffer_size, payload_size - len(payload)))
                            if not chunk:
                                break
                        except socket.timeout:
                            logger.log_error(f"Client '{client_data.name}' timeout!")
                            continue
                        except ConnectionResetError:
                            logger.log_error(f"Client '{client_data.name}' disconnected unexpectedly!")
                            client_is_alive = False
                            break

                        payload += chunk

                    data = pickle.loads(bytes(payload))
                    self.server_evt_fn(COMM_EVT_MODEL, client_data, data)
                elif packet_type == COMM_HEADER_TYPES_NOTI:
                    logger.log_debug(f"Notification received from '{client_data.name}' (packet_param1={packet_param1}).")
                    if packet_param1 == COMM_HEADER_NOTI_EPOCH_DONE:
                        self.download_total_size += payload_size
                        self.download_data_size += payload_size
                        bin_data = client_data.connection.recv(payload_size)
                        self.server_evt_fn(COMM_EVT_EPOCH_DONE_NOTIFY, client_data, Common.data_convert_from_bytes(bin_data))
        client_data.connection.close()
        logger.log_debug(f"Connection closed...")
        del self.clients[client_data.name]
        self.server_evt_fn(COMM_EVT_DISCONNECTED, client_data, None)

    def get_clients(self):
        return self.clients
    
    def send_data(self, client_name, data):
        logger.log_debug(f"Sending data to '{client_name}'(size of data = {len(data_bytes_array)})")
        data_bytes_array = Common.data_convert_to_bytes(data)
        header_data   = struct.pack(COMM_HEADER_FORMAT, COMM_HEADER_SIGN, len(data_bytes_array), COMM_HEADER_TYPES_DATA, 0, 0)
        self.upload_total_size += len(header_data) + len(data_bytes_array)
        self.upload_data_size += len(data_bytes_array)
        self.clients[client_name].connection.sendall(header_data)
        self.clients[client_name].connection.sendall(data_bytes_array)

    def send_command(self, client_name, command, param, data = None):
        logger.log_debug(f"Sending command to '{client_name}'(command={command}, size of data = {len(data)})")
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
        logger.log_debug(f"Client '{client_name}' is disconnected...")
        self.server_evt_fn(COMM_EVT_DISCONNECTED, self.clients[client_name], None)

