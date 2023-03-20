import logging
import traceback
from copy import copy
from dataclasses import dataclass
from enum import Enum
from time import sleep
from typing import Optional

from mpi4py import MPI


class Requests(Enum):
    FORK = 1


@dataclass
class Fork:
    is_clean: bool = False


def philosopher(
        philosopher_id: int,
        comm: MPI.Comm,
        left_fork: Optional[Fork],
        right_fork: Optional[Fork]
) -> None:
    cluster_size = comm.Get_size()

    right_philosopher = philosopher_id - 1
    if right_philosopher < 0:
        right_philosopher = cluster_size - 1
    left_philosopher = (philosopher_id + 1) % cluster_size

    send_left_fork = False
    send_right_fork = False

    has_sent_left = False
    has_sent_right = False

    logging.info(f"me: {philosopher_id}, left: {left_philosopher}, right: {right_philosopher}")

    while True:
        start_time = MPI.Wtime()

        print(f"Philosopher {philosopher_id} is thinking.")
        logging.info(f"Philosopher {philosopher_id} is thinking.")

        while (MPI.Wtime() - start_time) < 2:
            sleep(0.1)
            if comm.iprobe(left_philosopher):
                message = comm.recv(source=left_philosopher)

                if message == Requests.FORK and left_fork:
                    left_fork.is_clean = True
                    comm.send(left_fork, left_philosopher)
                    print('\t' * philosopher_id + f"Philosopher {philosopher_id} sent his left fork to philosopher {left_philosopher}")
                    logging.info(f"Philosopher {philosopher_id} sent his left fork to philosopher {left_philosopher}")
                    left_fork = None

            if comm.iprobe(right_philosopher):
                message = comm.recv(source=right_philosopher)

                if message == Requests.FORK and right_fork:
                    right_fork.is_clean = True
                    comm.send(right_fork, right_philosopher)
                    print('\t' * philosopher_id + f"Philosopher {philosopher_id} sent his right fork to philosopher {right_philosopher}")
                    logging.info(f"Philosopher {philosopher_id} sent his right fork to philosopher {right_philosopher}")
                    right_fork = None

        print('\t' * philosopher_id + f"Philosopher {philosopher_id} has finished thinking.")
        logging.info(f"Philosopher {philosopher_id} has finished thinking.")

        while not (left_fork and right_fork):
            if not left_fork:  # Send a request for the left fork if there is no left fork
                comm.send(Requests.FORK, left_philosopher)
                if not has_sent_left:
                    print(
                        '\t' * philosopher_id + f"Philosopher {philosopher_id} sent a request for his left fork to philosopher {left_philosopher}.")
                    logging.info(
                        f"Philosopher {philosopher_id} sent a request for his left fork to philosopher {left_philosopher}.")
                has_sent_left = True

            if not right_fork:  # Send a request for the right fork if there is no right fork
                comm.send(Requests.FORK, right_philosopher)
                if not has_sent_right:
                    print(
                        '\t' * philosopher_id + f"Philosopher {philosopher_id} sent a request for his right fork to philosopher {right_philosopher}.")
                    logging.info(
                        f"Philosopher {philosopher_id} sent a request for his right fork to philosopher {right_philosopher}.")
                has_sent_right = True

            if comm.iprobe(source=left_philosopher):
                message = comm.recv(source=left_philosopher)

                if message == Requests.FORK and left_fork and not left_fork.is_clean:
                    left_fork.is_clean = True
                    comm.send(left_fork, left_philosopher)
                    print('\t' * philosopher_id + f"Philosopher {philosopher_id} sent his left fork to philosopher {left_philosopher}")
                    logging.info(f"Philosopher {philosopher_id} sent his left fork to philosopher {left_philosopher}")
                    left_fork = None

                if message == Requests.FORK and left_fork and left_fork.is_clean:
                    send_left_fork = True

                if isinstance(message, Fork):
                    left_fork = copy(message)
                    print('\t' * philosopher_id + f"Philosopher {philosopher_id} received his left fork.")
                    logging.info(f"Philosopher {philosopher_id} received his left fork.")

            if comm.iprobe(source=right_philosopher):
                message = comm.recv(source=right_philosopher)

                if message == Requests.FORK and right_fork and not right_fork.is_clean:
                    right_fork.is_clean = True
                    comm.send(right_fork, left_philosopher)
                    print('\t' * philosopher_id + f"Philosopher {philosopher_id} sent his right fork to philosopher {right_philosopher}")
                    logging.info(f"Philosopher {philosopher_id} sent his right fork to philosopher {right_philosopher}")
                    right_fork = None

                if message == Requests.FORK and right_fork and right_fork.is_clean:
                    send_right_fork = True

                if isinstance(message, Fork):
                    right_fork = copy(message)
                    print('\t' * philosopher_id + f"Philosopher {philosopher_id} received his right fork.")
                    logging.info(f"Philosopher {philosopher_id} received his right fork.")

        print('\t' * philosopher_id + f"Philosopher {philosopher_id} is eating.")
        logging.info(f"Philosopher {philosopher_id} is eating.")
        start_time = MPI.Wtime()

        while (MPI.Wtime() - start_time) < 5:
            if comm.iprobe(left_philosopher):
                message = comm.recv(source=left_philosopher)

                if message == Requests.FORK and left_fork:
                    send_left_fork = True

            if comm.iprobe(right_philosopher):
                message = comm.recv(source=right_philosopher)

                if message == Requests.FORK and right_fork:
                    send_right_fork = True

        print('\t' * philosopher_id + f"Philosopher {philosopher_id} has finished eating.")
        logging.info(f"Philosopher {philosopher_id} has finished eating.")
        left_fork.is_clean = False

        logging.info(f"{send_left_fork = }, {send_right_fork = }")

        if send_left_fork:
            left_fork.is_clean = True
            comm.send(left_fork, left_philosopher)
            print('\t' * philosopher_id + f"Philosopher {philosopher_id} sent his left fork to philosopher {left_philosopher}")
            logging.info(f"Philosopher {philosopher_id} sent his left fork to philosopher {left_philosopher}")
            left_fork = None

        if send_right_fork:
            right_fork.is_clean = True
            comm.send(right_fork, right_philosopher)
            print('\t' * philosopher_id + f"Philosopher {philosopher_id} sent his right fork to philosopher {right_philosopher}")
            logging.info(f"Philosopher {philosopher_id} sent his right fork to philosopher {right_philosopher}")
            right_fork = None

        logging.info(f"Philosopher {philosopher_id} finished.")


def main():
    print("Started")
    comm = MPI.COMM_WORLD
    philosopher_id = comm.Get_rank()
    cluster_size = comm.Get_size()

    logging.basicConfig(
        filename=f"philosopher_{philosopher_id}.txt",
        filemode='w',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        level=logging.DEBUG
    )

    if philosopher_id == 0:
        philosopher(philosopher_id, comm, Fork(), Fork())
    elif philosopher_id == cluster_size - 1:
        philosopher(philosopher_id, comm, None, None)
    else:
        philosopher(philosopher_id, comm, Fork(), None)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.error(traceback.format_exc())
