
import pickle
import socket
import threading
from utils.consts import *
from utils.common import *
import struct
from utils.logger import *
import copy
from core.network import Network


class ServerComm(Network):

    def __init__(self, host, port, server_evt_fn):
        super().__init__()
        self.host = host
        self.port = port
        self.server_ready = threading.Lock()
        self.server_ready.acquire()
        server_thread = threading.Thread(target=self.__server_thread)
        self.clients = {}
        self.alive = True
        self.server_evt_fn = server_evt_fn

        server_thread.start()
        self.server_ready.acquire()  #Block the server until server network gets ready.

        mailbox_thread = threading.Thread(target=self.__client_receiver)
        mailbox_thread.start()
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
            self.no_rcvd_total += len(msg_header_bin)
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
        if result_dict["packet_sign"] == COMM_HEADER_SIGN:
            packet_type   = result_dict["type"]
            #packet_param1 = result_dict["param1"]
            #packet_param2 = result_dict["param2"]
            if packet_type == COMM_HEADER_TYPES_INTRODUCTION:
                client_info = connection.recv(COMM_HCHUNK_TOTAL_DATA_SIZE) #Read the payload
                self.no_rcvd_total += len(client_info)
                self.no_rcvd_data  += len(client_info)
                info = pickle.loads(client_info)
                client_data = ClientData(info["name"], address, info["processing_power"], connection)
                client_data.listener_thread = self.create_new_receiver(info["name"], connection)
                self.clients[info["name"]] = client_data
                client_name = info["name"]
                
                logger.log_debug(f"Client '{client_name}' on address '{address}' has been successfully introduced and added to the clients' pool.")
            else:
                logger.log_error(f"Client '{client_name}' on address '{address}' introduction failed (introduction needed but wrong command received)!")
                connection.close()
        else:
            logger.log_error(f"Client '{client_name}' on address '{address}' introduction error (packet_sign)!")
            connection.close()

    def __client_receiver(self):
        
        client_is_alive = True
        while(client_is_alive):
            packet_type, packet_param1, _, _, data, rcv_id = self.receive_data()
            client_data = self.clients[rcv_id]
            if packet_type == COMM_HEADER_TYPES_CMD:
                logger.log_debug(f"Command received from '{client_data.name}' (packet_param1={packet_param1}).")
                if packet_param1 == COMM_HEADER_CMD_NOP:
                    pass
                elif packet_param1 == COMM_HEADER_CMD_DROPEME_REQ:
                    self.server_evt_fn(COMM_EVT_DROPEME_REQ, client_data, None)
                    client_is_alive = False
                elif packet_param1 == COMM_HEADER_CMD_REQUEST_PACKET_NUM:
                    pass
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
                elif packet_param1 == COMM_HEADER_NOTI_TRAINNING_DONE:
                    self.server_evt_fn(COMM_EVT_TRAINING_DONE, client_data, None)
                else:
                     logger.log_error(f"Unknown notification received! (packet_param1={packet_param1})")
        client_data.connection.close()
        logger.log_debug(f"Connection closed...")
        del self.clients[client_data.name]
        self.server_evt_fn(COMM_EVT_DISCONNECTED, client_data, None)

    def get_clients(self):
        return self.clients
    
    def send_data_pkg(self, client_name, data):
        data_bytes_array = Common.data_convert_to_bytes(data)
        logger.log_debug(f"Sending data to '{client_name}'(size of data = {len(data_bytes_array)})")
        self.send_data(self.clients[client_name].connection, client_name, COMM_HEADER_TYPES_DATA, 0, 0, data_bytes_array)

    def send_command(self, client_name, command, param, data = None):
        logger.log_debug(f"Sending command to '{client_name}'(command={command}, size of data = {len(data)})")
        data_bytes_array = b''
        if data is not None:
            data_bytes_array = Common.data_convert_to_bytes(data)
        self.send_data(self.clients[client_name].connection, client_name, COMM_HEADER_TYPES_CMD, command, param, data_bytes_array)

    def disconnect(self, client_name):
        self.clients[client_name].connection.close()
        logger.log_debug(f"Client '{client_name}' is disconnected...")
        self.server_evt_fn(COMM_EVT_DISCONNECTED, self.clients[client_name], None)