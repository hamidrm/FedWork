import queue
import threading
import struct
import copy
import time
from utils.consts import *
from utils.common import *
from utils.logger import *

class Network:
    def __init__(self):
        self.recv_queue = queue.Queue(maxsize=10)

        self.send_queue = queue.Queue(maxsize=10)

        self.sending_thread = threading.Thread(target=self.__send_thread)
        self.sending_thread.start()

        self.receiving_threads = []

        self.thread_num = 0

    def create_new_receiver(self, recv_id, connection):
        thread = threading.Thread(target=self.__receive_data, args=(recv_id, connection, ))
        thread.start()
        self.receiving_threads.append(None)
        self.receiving_threads[self.thread_num] = thread
        self.thread_num += 1
        return thread

    def receive_data(self):
       # try:
        item = self.recv_queue.get(block=True)
        return (item["packet_type"], item["packet_param1"], item["packet_param2"], item["expected_length"], item["total_data"], item["rcv_id"])
        #finally:
        #    self.send_queue.task_done()

    def send_data(self, connection, receiver_id, type, param1, param2, data):
        item = {}
        item["connection"] = connection
        item["receiver_id"] = receiver_id
        item["type"] = type
        item["param1"] = param1
        item["param2"] = param2
        item["data"] = copy.deepcopy(data)

        while True:
            try:
                self.send_queue.put(item, block=True)
                break
            except queue.Full:
                logger.log_error("Send queue is full. Waiting...")
                time.sleep(0.5)

    def __send_thread(self):
        while True:
            try:
                item = self.send_queue.get()
                self.__send_data(item["connection"], item["receiver_id"], item["type"], item["param1"], item["param2"], item["data"])
            finally:
                self.send_queue.task_done()

    def resend_chunk_data(self, connection, receiver_id, chunk_seq_num):
        payload_offset = COMM_HCHUNK_TOTAL_DATA_SIZE+chunk_seq_num*COMM_TCHUNK_TOTAL_DATA_SIZE

        with self.send_buffer_lock:
            payload_length = min(COMM_TCHUNK_TOTAL_DATA_SIZE, len(self.send_buffer) - payload_offset)
            payload = self.send_buffer[payload_offset:payload_length+payload_offset]

        data_buffer = struct.pack(COMM_TAILS_FORMAT, COMM_TAILS_SIGN, receiver_id.encode('ascii'), payload_length, chunk_seq_num)
        
        data_pad = bytes([0 for _ in range(COMM_TCHUNK_TOTAL_DATA_SIZE - payload_length)])
        connection.sendall(data_buffer+payload+data_pad)

                    
    def __send_data(self, connection, receiver_id, type, param1, param2, data):
        data_offset = 0
        data_seq_num = 0
        
        payload_to_send = Common.data_convert_to_bytes(data)
        logger().log_debug(f"Sending {len(data)} bytes. rcv_id={receiver_id},  packet_type={hex(type)}, packet_param1={hex(param1)}")
        while True:

            if data_offset == 0:
                data_buffer = struct.pack(COMM_HEADER_FORMAT, COMM_HEADER_SIGN, len(payload_to_send), type, param1, param2, receiver_id.encode('ascii'))
                data_length = min(len(payload_to_send), COMM_HCHUNK_TOTAL_DATA_SIZE)
                data_to_send = payload_to_send[:data_length]
                data_offset += data_length
                data_buffer += data_to_send
                data_pad = bytes([0 for _ in range(COMM_HCHUNK_TOTAL_DATA_SIZE - data_length)])
                connection.sendall(data_buffer+data_pad)
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
                connection.sendall(data_buffer+data_pad)
                
                if data_offset == len(payload_to_send):
                    return

    def __add_to_recv_packet(self, packet_type, packet_param1, packet_param2, expected_length, total_data, rcv_id):
        item = {}
        item["packet_type"] = packet_type
        item["packet_param1"] = packet_param1
        item["packet_param2"] = packet_param2
        item["expected_length"] = expected_length
        item["expected_length"] = expected_length
        item["total_data"] = copy.deepcopy(total_data)
        item["rcv_id"] = rcv_id

        while True:
            try:
                self.recv_queue.put(item, block=True)
                break
            except queue.Full:
                logger.log_error("Recv queue is full!")

    def recvall(self, sock, expected_length):
        received_data = b''
        while len(received_data) < expected_length:
            remaining_length = expected_length - len(received_data)
            data_chunk = sock.recv(remaining_length)
            if not data_chunk:
                raise ConnectionError("Connection closed by peer")
            received_data += data_chunk
        return received_data

    def __receive_data(self, recv_id, connection):
        expected_length = 0
        received_length = 0
        total_data = []
        seq_list = []


        while True:
            chunk = self.recvall(connection, COMM_CHUNK_TOTAL_SIZE)

            header_data   = struct.unpack(COMM_HEADER_FORMAT, chunk[:COMM_HEADER_SIZE])
            header_dict = dict(zip(COMM_HEADER_DICT.keys(), header_data)) 

            if header_dict["packet_sign"] == COMM_HEADER_SIGN:
                payload_size  = header_dict["payload_len"]
                packet_type   = header_dict["type"]
                packet_param1 = header_dict["param1"]
                packet_param2 = header_dict["param2"]
                packet_id     = header_dict["id"].decode('ascii').rstrip('\x00')
                logger.log_debug(f"receiving {payload_size} bytes. rcv_id={packet_id}, packet_type={hex(packet_type)}, packet_param1={hex(packet_param1)}")
                if packet_id != recv_id:
                    logger.log_warning(f"Invalid chunk is read! expected '{recv_id}', received '{packet_id}'.")
                    continue

                expected_length = payload_size
                
                received_length = min(payload_size, COMM_HCHUNK_TOTAL_DATA_SIZE)

                total_data = [0] * expected_length

                total_data[:payload_size] = chunk[COMM_HEADER_SIZE:COMM_HEADER_SIZE+received_length]

                if expected_length <= COMM_HCHUNK_TOTAL_DATA_SIZE:
                    self.__add_to_recv_packet(packet_type, packet_param1, packet_param2, expected_length, bytes(total_data), recv_id)
                
            elif header_dict["packet_sign"] == COMM_TAILS_SIGN:
                tails_data   = struct.unpack(COMM_TAILS_FORMAT, chunk[:COMM_TAILS_SIZE])
                tails_dict = dict(zip(COMM_TAILS_DICT.keys(), tails_data))

                payload_size  = tails_dict["payload_len"]
                packet_seq   = tails_dict["sequence"]
                packet_id     = tails_dict["id"].decode('ascii').rstrip('\x00')

                if packet_id != recv_id:
                    logger().log_warning(f"Invalid chunk is read! expected: '{recv_id}', received: '{packet_id}'.")
                    continue
                
                seq_list.append(packet_seq)

                total_data[packet_seq * COMM_TCHUNK_TOTAL_DATA_SIZE + COMM_HCHUNK_TOTAL_DATA_SIZE:packet_seq * COMM_TCHUNK_TOTAL_DATA_SIZE + COMM_HCHUNK_TOTAL_DATA_SIZE + payload_size] = chunk[COMM_TAILS_SIZE:COMM_TAILS_SIZE+payload_size]
                received_length += payload_size

                if packet_seq == int((expected_length - COMM_HCHUNK_TOTAL_DATA_SIZE - 1) / COMM_TCHUNK_TOTAL_DATA_SIZE):
                    if received_length == expected_length:
                        logger.log_debug(f"{len(total_data)} bytes have received. rcv_id={packet_id}, packet_type={hex(packet_type)}, packet_param1={hex(packet_param1)}")
                        self.__add_to_recv_packet(packet_type, packet_param1, packet_param2, expected_length, bytes(total_data), recv_id)
                    else:
                        logger().log_error(f"We faced such a critical and vital problem!'.")
                        # expected_seq_list = [i for i in range(int((expected_length - COMM_HCHUNK_TOTAL_DATA_SIZE) / COMM_TCHUNK_TOTAL_DATA_SIZE)) if i not in seq_list]
                        # # We have lost some packets :(
                        # # We must request resend them
                        # for packet_num in expected_seq_list:
                        #     while True:
                        #         self.send_command(recv_id, COMM_HEADER_CMD_REQUEST_PACKET_NUM, packet_num, None)
                        #         chunk = connection.recv(COMM_CHUNK_TOTAL_SIZE)
                                
                        #         tails_data   = struct.unpack(COMM_TAILS_FORMAT, chunk[:COMM_TAILS_SIZE])
                        #         tails_dict = dict(zip(COMM_TAILS_DICT.keys(), tails_data))

                        #         payload_size  = tails_dict["payload_len"]
                        #         packet_seq   = tails_dict["sequence"]
                        #         packet_id     = tails_dict["id"].decode('ascii').rstrip('\x00')

                        #         if packet_id != recv_id:
                        #             logger().log_warning(f"Invalid chunk is read! expected: '{recv_id}', received: '{packet_id}'.")
                        #             continue

                        #         if packet_seq != packet_num:
                        #             logger().log_warning(f"Invalid packet chunk received! expected seq: '{packet_num}', received: '{packet_seq}'.")
                        #             continue

                        #         total_data[packet_seq * COMM_TCHUNK_TOTAL_DATA_SIZE + COMM_HCHUNK_TOTAL_DATA_SIZE:packet_seq * COMM_TCHUNK_TOTAL_DATA_SIZE + COMM_HCHUNK_TOTAL_DATA_SIZE + payload_size] = chunk[COMM_TAILS_SIZE:COMM_TAILS_SIZE+payload_size]
                        #         received_length += payload_size
                        # return (packet_type, packet_param1, packet_param2, expected_length, bytes(total_data))