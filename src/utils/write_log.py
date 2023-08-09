import sys
import os
import time


# 控制台输出记录到文件
class Logger(object):
    def __init__(self, log_path, args, method, stream=sys.stdout):
        self.log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + "_{}_{}_{}".format(args.dataset,
                                                                                                           args.model, method) + '.log'
        self.terminal = stream
        self.log = open(self.log_file_name, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    # 自定义目录存放日志文
    pass
