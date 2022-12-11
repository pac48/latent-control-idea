import threading

import numpy as np
from ZMQ_server import ZMQServer

server = ZMQServer(5555, 'numpy')


def send_data():
    while True:
        dims_str = input("enter size")
        # dims_str = '2 2'
        dims = [int(x) for x in dims_str.split()]
        array = np.random.random((dims))
        print(array)
        server.send(array)


def recv_data():
    while True:
        arr = server.recv()
        print(arr)


if __name__ == "__main__":
    # creating thread
    t1 = threading.Thread(target=send_data)
    t2 = threading.Thread(target=recv_data)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print("Done!")
