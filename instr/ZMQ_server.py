import numpy as np
import zmq
from proto.msgs_pb2 import FLOAT64, INT32, UINT8
from proto.msgs_pb2 import Numpy as NumpyMsg


class ZMQServer:
    def __init__(self, port, topic):
        self.topic = topic
        self.port = port

        self.recv_context = zmq.Context()
        self.recv_socket = self.recv_context.socket(zmq.SUB)
        self.recv_socket.connect("tcp://localhost:" + str(port))
        self.recv_socket.setsockopt(zmq.SUBSCRIBE, topic.encode())

        self.send_context = zmq.Context()
        self.send_socket = self.send_context.socket(zmq.PUB)
        self.send_socket.connect("tcp://localhost:" + str(port + 1))
        self.send_socket.connect("tcp://localhost:" + str(port + 1))
        self.send_socket.connect("tcp://localhost:" + str(port + 1))

    def recv(self):
        """
        block until new message arrives
        :return:
        """
        recv_topic = self.recv_socket.recv_string()  # flags=zmq.NOBLOCK
        msg = NumpyMsg()
        data = self.recv_socket.recv()
        msg.ParseFromString(data)
        if msg.type == FLOAT64:
            arr = np.frombuffer(msg.data, dtype=np.float)
        elif msg.type == INT32:
            arr = np.frombuffer(msg.data, dtype=np.int32)
        elif msg.type == UINT8:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
        else:
            raise NotImplementedError

        arr = np.reshape(arr, tuple(d for d in msg.dims), order='F')

        return arr

    def send(self, arr):
        """
        send arr to Matlab
        :return:
        """
        self.send_socket.send_string(self.topic, zmq.SNDMORE)
        msg = NumpyMsg()
        for d in arr.shape:
            msg.dims.append(d)
        msg.dims.reverse()
        msg.num_dims = len(arr.shape)
        if arr.dtype == np.float64:
            msg.type = FLOAT64
        elif arr.dtype == np.float32:
            arr = np.array(arr, dtype=np.float64)
            msg.type = FLOAT64
        elif arr.dtype == np.int32:
            msg.type = INT32
        elif arr.dtype == np.uint8:
            msg.type = UINT8
        elif arr.dtype == np.int64:
            msg.type = INT32
            arr = np.array(arr, dtype=np.int32)
        else:
            raise NotImplementedError

        msg.data = arr.tobytes()

        self.send_socket.send(msg.SerializeToString())
