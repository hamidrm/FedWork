import queue
import sys
import threading
import copy
import time

class DummyNetwork:
    def __init__(self):
        self.sending_mutex = threading.Lock()

        self.recv_queue = queue.Queue(maxsize=100)  # Larger buffer, faster
        self.send_queue = queue.Queue(maxsize=100)

        self.send_thread_stop = threading.Event()
        self.sending_thread = threading.Thread(target=self.__send_thread)
        self.sending_thread.start()

        self.recv_thread_stop = []
        self.receiving_threads = []

        self.thread_num = 0

        self.no_sent_data = 0
        self.no_rcvd_data = 0

        self.no_sent_total = 0
        self.no_rcvd_total = 0
        
    def create_new_receiver(self, recv_id, connection=None):
        stop_event = threading.Event()
        self.recv_thread_stop.append(stop_event)
        thread = threading.Thread(target=self.__receive_data, args=(recv_id, stop_event))
        thread.start()
        self.receiving_threads.append((thread, stop_event))
        self.thread_num += 1
        return thread

    def send_data(self, connection, receiver_id, type, param1, param2, data):
        item = {
            "receiver_id": receiver_id,
            "type": type,
            "param1": param1,
            "param2": param2,
            "data": copy.deepcopy(data),
        }
        while True:
            try:
                self.send_queue.put(item, block=True)
                break
            except queue.Full:
                time.sleep(0.01)

    def receive_data(self):
        item = self.recv_queue.get(block=True)
        if item is None:
            return None, None, None, None, None, None
        return (item["packet_type"], item["packet_param1"], item["packet_param2"], item["expected_length"], item["total_data"], item["rcv_id"])

    def __send_thread(self):
        while not self.send_thread_stop.is_set():
            try:
                item = self.send_queue.get()
                if item is not None:
                    with self.sending_mutex:
                        self.__send_data(item)
            finally:
                self.send_queue.task_done()

    def __send_data(self, item):
        payload_to_send = item["data"]
        receiver_id = item["receiver_id"]
        self.no_sent_total += sys.getsizeof(item)
        self.no_sent_data += len(payload_to_send)

        # Simulate sending a packet: push directly into a recv queue
        packet = {
            "packet_type": item["type"],
            "packet_param1": item["param1"],
            "packet_param2": item["param2"],
            "expected_length": len(payload_to_send),
            "total_data": copy.deepcopy(payload_to_send),
            "rcv_id": receiver_id,
        }
        self.recv_queue.put(packet)

    def __receive_data(self, recv_id, stop_event):
        while not stop_event.is_set():
            try:
                packet = self.recv_queue.get(timeout=0.1)
                self.no_rcvd_total += sys.getsizeof(packet)
                if packet["rcv_id"] == recv_id:
                    self.no_rcvd_data += len(packet["total_data"])
                    self.recv_queue.put(packet)  # put it back for receive_data() to pick
            except queue.Empty:
                continue

    def stop(self):
        self.send_thread_stop.set()
        self.sending_thread.join()

        for thread, stop_event in self.receiving_threads:
            stop_event.set()
            thread.join()
