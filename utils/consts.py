MEASURE_PROBE_AGGR_TIME = "AggregationTime"
MEASURE_PROBE_EVAL_TIME = "EvaluationTime"
MEASURE_PROBE_EVAL_ACC = "EvaluationAccuracy"
MEASURE_PROBE_EVAL_LOSS = "EvaluationLoss"
MEASURE_PROBE_CLIENT_ACC = "AccuracyOfClient#"
MEASURE_PROBE_CLIENT_LOSS = "LossOfClient#"
MEASURE_PROBE_TOTAL_RCVD_BYTES = "ServerTotalRecvBytes"
MEASURE_PROBE_TOTAL_SENT_BYTES = "ServerTotalSentBytes"
MEASURE_PROBE_DATA_RCVD_BYTES = "ServerDataRecvBytes"
MEASURE_PROBE_DATA_SENT_BYTES = "ServerDataSentBytes"

OUTPUT_DIR = "output"

COMM_CHUNK_TOTAL_SIZE = 4096

SERVER_NAME = "*SERVER*"

COMM_HEADER_SIZE = 24
COMM_HCHUNK_TOTAL_DATA_SIZE = (COMM_CHUNK_TOTAL_SIZE - COMM_HEADER_SIZE)
COMM_HEADER_FORMAT = "!IIIhh8s"
COMM_HEADER_DICT = {"packet_sign" : 0, "payload_len" : 0, "type" : 0, "param1" : 0, "param2" : 0, "id" : 0}


COMM_TAILS_SIZE = 18
COMM_TCHUNK_TOTAL_DATA_SIZE = (COMM_CHUNK_TOTAL_SIZE - COMM_TAILS_SIZE)
COMM_TAILS_FORMAT = "!I8sIh"
COMM_TAILS_DICT = {"packet_sign" : 0, "id" : 0, "payload_len" : 0, "sequence" : 0}

COMM_HEADER_SIGN = 0xFEDAFEDA
COMM_TAILS_SIGN = (COMM_HEADER_SIGN + 1)

COMM_HEADER_TYPES_CMD = 0x80000000
COMM_HEADER_TYPES_DATA = 0x80000001
COMM_HEADER_TYPES_INTRODUCTION = 0x80000002
COMM_HEADER_TYPES_NOTI = 0x80000003

COMM_HEADER_CMD_NOP = 0x0000
COMM_HEADER_CMD_TURNOFF = 0x0001
COMM_HEADER_CMD_START_TRAINNING = 0x0002
COMM_HEADER_CMD_TRAINNING_DONE = 0x0003
COMM_HEADER_CMD_DROPEME_REQ = 0x0004
COMM_HEADER_CMD_GET_TOTAL_EPOCHS = 0x0005
COMM_HEADER_CMD_GET_TRAINING_COUNT = 0x0006
COMM_HEADER_CMD_START_PERIODIC_MODE = 0x0007
COMM_HEADER_CMD_STOP_PERIODIC_MODE = 0x0008
COMM_HEADER_CMD_REQUEST_PACKET_NUM = 0x0009

COMM_HEADER_NOTI_EPOCH_DONE = 0x0001

COMM_EVT_TRAINING_START = 0x80
COMM_EVT_TRAINING_DONE = 0x81
COMM_EVT_DROPEME_REQ = 0x82
COMM_EVT_MODEL = 0x83
COMM_EVT_CONNECTED = 0x84
COMM_EVT_DISCONNECTED = 0x85
COMM_EVT_TURNOFF = 0x86
COMM_EVT_EPOCH_DONE_NOTIFY = 0x87
COMM_EVT_EPOCHS_TOTAL_COUNT_REQ = 0x88
COMM_EVT_TRAINING_TOTAL_COUNT_REQ = 0x89
COMM_EVT_START_PERIODIC_TRAINING = 0x8A
COMM_EVT_STOP_PERIODIC_TRAINING = 0x8B