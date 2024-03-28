import socket
import threading
import struct
import copy
from utils.consts import *
from utils.common import *
from utils.logger import *

class Network:
    def __init__(self, name):
        self.send_buffer = []
        self.recv_lock = threading.Lock()
        self.send_buffer_lock = threading.Lock()
        self.name = name


    def resend_chunk_data(self, connection, receiver_id, chunk_seq_num):
        payload_offset = COMM_HCHUNK_TOTAL_DATA_SIZE+chunk_seq_num*COMM_TCHUNK_TOTAL_DATA_SIZE
        with self.send_buffer_lock:
            payload_length = min(COMM_TCHUNK_TOTAL_DATA_SIZE, len(self.send_buffer) - payload_offset)
            payload = self.send_buffer[payload_offset:payload_length+payload_offset]

        data_buffer = struct.pack(COMM_TAILS_FORMAT, COMM_TAILS_SIGN, receiver_id.encode('ascii'), payload_length, chunk_seq_num)
        
        data_pad = bytes([0 for _ in range(COMM_TCHUNK_TOTAL_DATA_SIZE - payload_length)])
        connection.sendall(data_buffer+payload+data_pad)


                    
    def send_data(self, connection, receiver_id, type, param1, param2, data):
        data_offset = 0
        data_seq_num = 0
        

        payload_to_send = data
        with self.send_buffer_lock:
            self.send_buffer = copy.deepcopy(payload_to_send)

        logger().log_debug(f"Going to send {len(payload_to_send)} bytes, recv_id={receiver_id}, type={type}, param1={param1}, param2={param2}")
        while True:

            if data_offset == 0:
                data_buffer = struct.pack(COMM_HEADER_FORMAT, COMM_HEADER_SIGN, len(payload_to_send), type, param1, param2, receiver_id.encode('ascii'))
                data_length = min(len(payload_to_send), COMM_HCHUNK_TOTAL_DATA_SIZE)
                data_to_send = payload_to_send[:data_length]
                data_offset += data_length
                data_buffer += data_to_send
                data_pad = bytes([0 for _ in range(COMM_HCHUNK_TOTAL_DATA_SIZE - data_length)])
                connection.sendall(data_buffer+data_pad)
                logger().log_debug(f"Initial packet is sent. len={len(payload_to_send)}, recv_id={receiver_id}, type={type}, param1={param1}, param2={param2}")
                if len(payload_to_send) <= COMM_HCHUNK_TOTAL_DATA_SIZE:
                    logger().log_debug(f"{len(payload_to_send)} bytes sent, recv_id={receiver_id}, type={type}, param1={param1}, param2={param2}")
                    return
            else:
                data_length = min(len(payload_to_send) - data_offset, COMM_TCHUNK_TOTAL_DATA_SIZE)
                data_buffer = struct.pack(COMM_TAILS_FORMAT, COMM_TAILS_SIGN, receiver_id.encode('ascii'), data_length, data_seq_num)
                data_seq_num += 1
                data_to_send = payload_to_send[data_offset:data_offset+data_length]
                data_offset += data_length
                data_buffer += data_to_send
                data_pad = bytes([0 for _ in range(COMM_TCHUNK_TOTAL_DATA_SIZE - data_length)])
                connection.sendall(data_buffer+data_pad)
                logger().log_debug(f"Packet '{data_seq_num}-th' is sent. len={len(payload_to_send)}, recv_id={receiver_id}, type={type}, param1={param1}, param2={param2}")
                if data_offset == len(payload_to_send):
                    logger().log_debug(f"{len(payload_to_send)} bytes sent, recv_id={receiver_id}, type={type}, param1={param1}, param2={param2}")
                    return

    def receive_data(self, recv_id, connection):
        expected_length = 0
        received_length = 0
        total_data = []
        seq_list = []

        while True:
            chunk = connection.recv(COMM_CHUNK_TOTAL_SIZE)
            logger().log_debug(f"'{len(chunk)}' bytes has been received. name={self.name}, recv_id={recv_id}")
            header_data   = struct.unpack(COMM_HEADER_FORMAT, chunk[:COMM_HEADER_SIZE])
            header_dict = dict(zip(COMM_HEADER_DICT.keys(), header_data)) 

            if header_dict["packet_sign"] == COMM_HEADER_SIGN:
                payload_size  = header_dict["payload_len"]
                packet_type   = header_dict["type"]
                packet_param1 = header_dict["param1"]
                packet_param2 = header_dict["param2"]
                packet_id     = header_dict["id"].decode('ascii').rstrip('\x00')
                logger().log_debug(f"New initialize packet is received (payload_size={payload_size}, packet_type={packet_type}, packet_param1={packet_param1}, packet_param2={packet_param2}, packet_id={packet_id}). name={self.name}, recv_id={recv_id}")
                if packet_id != recv_id:
                    logger().log_warning(f"Invalid chunk is read! expected '{recv_id}', received '{packet_id}'.")
                    continue

                expected_length = payload_size
                
                received_length = min(payload_size, COMM_HCHUNK_TOTAL_DATA_SIZE)

                total_data = [0] * expected_length

                total_data[:payload_size] = chunk[COMM_HEADER_SIZE:COMM_HEADER_SIZE+payload_size]

                if expected_length <= COMM_HCHUNK_TOTAL_DATA_SIZE:
                    return (packet_type, packet_param1, packet_param2, expected_length, bytes(total_data[:expected_length]))
                
            elif header_dict["packet_sign"] == COMM_TAILS_SIGN:
                tails_data   = struct.unpack(COMM_TAILS_FORMAT, chunk[:COMM_TAILS_SIZE])
                tails_dict = dict(zip(COMM_TAILS_DICT.keys(), tails_data))

                payload_size  = tails_dict["payload_len"]
                packet_seq   = tails_dict["sequence"]
                packet_id     = tails_dict["id"].decode('ascii').rstrip('\x00')

                logger().log_debug(f"New tail packet is received (payload_size={payload_size}, packet_seq={packet_seq}, packet_id={packet_id}")
                
                if packet_id != recv_id:
                    logger().log_warning(f"Invalid chunk is read! expected: '{recv_id}', received: '{packet_id}'.")
                    continue
                
                seq_list.append(packet_seq)

                total_data[packet_seq * COMM_TCHUNK_TOTAL_DATA_SIZE + COMM_HCHUNK_TOTAL_DATA_SIZE:packet_seq * COMM_TCHUNK_TOTAL_DATA_SIZE + COMM_HCHUNK_TOTAL_DATA_SIZE + payload_size] = chunk[COMM_TAILS_SIZE:COMM_TAILS_SIZE+payload_size]

                received_length += payload_size

                if packet_seq == int((expected_length - COMM_HCHUNK_TOTAL_DATA_SIZE - 1) / COMM_TCHUNK_TOTAL_DATA_SIZE):
                    if received_length == expected_length:
                        return (packet_type, packet_param1, packet_param2, expected_length, bytes(total_data[:expected_length]))
                    else:
                        expected_seq_list = [i for i in range(int((expected_length - COMM_HCHUNK_TOTAL_DATA_SIZE) / COMM_TCHUNK_TOTAL_DATA_SIZE)) if i not in seq_list]
                        # We have lost some packets :(
                        # We must request resend them
                        for packet_num in expected_seq_list:
                            while True:
                                self.send_command(recv_id, COMM_HEADER_CMD_REQUEST_PACKET_NUM, packet_num, None)
                                chunk = connection.recv(COMM_CHUNK_TOTAL_SIZE)
                                
                                tails_data   = struct.unpack(COMM_TAILS_FORMAT, chunk[:COMM_TAILS_SIZE])
                                tails_dict = dict(zip(COMM_TAILS_DICT.keys(), tails_data))

                                if COMM_TAILS_SIGN != tails_dict["packet_sign"]:
                                    continue

                                payload_size  = tails_dict["payload_len"]
                                packet_seq   = tails_dict["sequence"]
                                packet_id     = tails_dict["id"].decode('ascii').rstrip('\x00')

                                if packet_id != recv_id:
                                    logger().log_warning(f"Invalid chunk is read! expected: '{recv_id}', received: '{packet_id}'.")
                                    continue

                                if packet_seq != packet_num:
                                    logger().log_warning(f"Invalid packet chunk received! expected seq: '{packet_num}', received: '{packet_seq}'.")
                                    continue

                                total_data[packet_seq * COMM_TCHUNK_TOTAL_DATA_SIZE + COMM_HCHUNK_TOTAL_DATA_SIZE:packet_seq * COMM_TCHUNK_TOTAL_DATA_SIZE + COMM_HCHUNK_TOTAL_DATA_SIZE + payload_size] = chunk[COMM_TAILS_SIZE:COMM_TAILS_SIZE+payload_size]
                                received_length += payload_size
                        return (packet_type, packet_param1, packet_param2, expected_length, bytes(total_data[:expected_length]))
            else:
                hex_sign = hex(header_dict["packet_sign"])
                logger().log_error(f"Invalid packet sign received! (packet_sign={hex_sign})")