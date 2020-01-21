# -*-coding:utf-8-*-

import threading
import time

exitFlag = 0


class MyThread (threading.Thread):
    def __init__(self, thread_id, name, counter):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.name = name
        self.counter = counter

    def run(self):
        print("" + self.name)
        print_time(self.name, self.counter, 5)
        print("" + self.name)


def print_time(thread_name, delay, counter):
    while counter:
        if exitFlag:
            thread_name.exit()
        time.sleep(delay)
        print("%s: %s" % (thread_name, time.ctime(time.time())))
        counter -= 1


# 创建新线程
thread1 = MyThread(1, "Thread-1", 0.1)
thread2 = MyThread(2, "Thread-2", 0.2)

print(threading.activeCount())
# 开启新线程
thread1.start()
print(threading.current_thread())
print(threading.enumerate())
thread2.start()
print(threading.activeCount())
thread1.join()
print(threading.activeCount())
thread2.join()
print(threading.activeCount())
print(threading.current_thread())
