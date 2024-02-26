COMM_HEADER_SIZE = 16
COMM_HEADER_FORMAT = "!IIIhh"
COMM_HEADER_DICT = {"packet_sign" : 0, "payload_len" : 0, "type" : 0, "param1" : 0, "param2" : 0}
COMM_HEADER_SIGN = 0xFEDAFEDA
COMM_HEADER_TYPES_CMD = 0x80000000
COMM_HEADER_TYPES_DATA = 0x80000001
COMM_HEADER_TYPES_INTRODUCTION = 0x80000002
COMM_HEADER_TYPES_NOTI = 0x80000003

COMM_HEADER_CMD_NOP = 0x0000
COMM_HEADER_CMD_TURNOFF = 0x0001
COMM_HEADER_CMD_START_TRAINNING = 0x0002
COMM_HEADER_CMD_TRAINNING_DONE = 0x0003
COMM_HEADER_CMD_DROPEME_REQ = 0x0004


COMM_HEADER_NOTI_EPOCH_DONE = 0x0001

COMM_EVT_TRAINING_START = 0x80
COMM_EVT_TRAINING_DONE = 0x81
COMM_EVT_DROPEME_REQ = 0x82
COMM_EVT_MODEL = 0x83
COMM_EVT_CONNECTED = 0x84
COMM_EVT_DISCONNECTED = 0x85
COMM_EVT_TURNOFF = 0x86
COMM_EVT_EPOCH_DONE_NOTIFY = 0x87