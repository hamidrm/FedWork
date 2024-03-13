from datetime import datetime
import enum
import os
import socket
from io import StringIO
import utils.consts
import utils.common

class logger_stdout_type(enum):
    logger_stdout_console = 1
    logger_stdout_file = 2
    logger_stdout_network = 4
    logger_stdout_stringio = 8
    
class logger_log_type(enum):
    logger_type_normal = 1
    logger_type_debug = 2
    logger_type_info = 4
    logger_type_warning = 8
    logger_type_error = 16
    
    

class logger:
    _class_ins = None

    def __new__(cls):
        if cls._class_ins is None:
            cls._class_ins = super().__new__(cls)
        return cls._class_ins
    
    def __init__(self):
        self.stdout_type_mask = logger_stdout_type.logger_stdout_console
        self.tag = ""
        self.file_name = f"output_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}_.log"
        self.log_file_path = os.path.join(os.getcwd(), utils.consts.OUTPUT_DIR)
        self.log_file_path = os.path.join(self.log_file_path, self.file_name)
        self.log_server_ip = "127.0.0.1"
        self.log_server_port = 6650
        self.log_server_socket = None
        self.log_stringio = None
        self.log_type_mask = 31

        self.cosnole_color_white = '\033[97m'
        self.cosnole_color_red = '\033[91m'
        self.cosnole_color_green = '\033[92m'
        self.cosnole_color_blue = '\033[94m'
        self.cosnole_color_orange = '\033[93m'
 

    def set_string_io(self, stringio_var: StringIO):
        self.log_stringio = stringio_var
    
    def set_server(self, server_addr: utils.common.IpAddr):
        self.log_server_ip = server_addr.get_ip()
        self.log_server_port = server_addr.get_port()
    
    def set_file_name(self, file_name):
        self.file_name = f"{file_name}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}_.log"
        self.log_file_path = os.path.join(os.getcwd(), utils.consts.OUTPUT_DIR)
        self.log_file_path = os.path.join(self.log_file_path, self.file_name)

    def set_tag(self, tag: str):
        self.tag = tag

    def set_stdout(self, stdout_flags: int):
        self.stdout_type_mask = stdout_flags

    def set_log_type(self, type_flags: int):
        self.log_type_mask = type_flags

    def log(self, msg: str, type: logger_log_type):
        if self.log_type_mask & int(type) == 0:
            return
        
        log_text = ""

        log_text = f"[{datetime.now().strftime("%H:%M:%S")}]"
        if self.tag != "":
            log_text = f"{log_text}[{self.tag}]"
        
        log_text += msg

        if (self.stdout_type_mask & logger_stdout_type.logger_stdout_console) != 0:
            painted_text = log_text

            if type == logger_log_type.logger_type_normal:
                painted_text = self.cosnole_color_white + painted_text
            elif type == logger_log_type.logger_type_info:
                painted_text = self.cosnole_color_green + painted_text
            elif type == logger_log_type.logger_type_debug:
                painted_text = self.cosnole_color_blue + painted_text
            elif type == logger_log_type.logger_type_warning:
                painted_text = self.cosnole_color_orange + painted_text
            elif type == logger_log_type.logger_type_error:
                painted_text = self.cosnole_color_red + painted_text

            print(painted_text)

        if (self.stdout_type_mask & logger_stdout_type.logger_stdout_file) != 0:
            with open(self.log_file_path, "a") as log_file:
                final_text = log_text

                if type == logger_log_type.logger_type_normal:
                    final_text = "[NORMAL]" + final_text
                elif type == logger_log_type.logger_type_info:
                    final_text = "[INFO]" + final_text
                elif type == logger_log_type.logger_type_debug:
                    final_text = "[DEBUG]" + final_text
                elif type == logger_log_type.logger_type_warning:
                    final_text = "[WARNING]" + final_text
                elif type == logger_log_type.logger_type_error:
                    final_text = "[ERROR]" + final_text
                
                log_file.write(final_text)

        if (self.stdout_type_mask & logger_stdout_type.logger_stdout_network) != 0:
            if self.log_server_socket == None:
                self.log_server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            self.log_server_socket.sendto((type, log_text), (self.log_server_ip, self.log_server_port))

        if (self.stdout_type_mask & logger_stdout_type.logger_stdout_stringio) != 0:
            if self.log_stringio == None:
                self.log_stringio = StringIO("")

            final_text = log_text

            if type == logger_log_type.logger_type_normal:
                final_text = "[NORMAL]" + final_text
            elif type == logger_log_type.logger_type_info:
                final_text = "[INFO]" + final_text
            elif type == logger_log_type.logger_type_debug:
                final_text = "[DEBUG]" + final_text
            elif type == logger_log_type.logger_type_warning:
                final_text = "[WARNING]" + final_text
            elif type == logger_log_type.logger_type_error:
                final_text = "[ERROR]" + final_text

            self.log_stringio.write(final_text)