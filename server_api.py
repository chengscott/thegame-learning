import os, signal, sys
import subprocess


class Server:
  def __init__(self, server='./thegame-server', port=55666):
    self.cmd = [server, '-listen', ':' + str(port)]

  def start(self):
    self.proc = subprocess.Popen(
        self.cmd, stdin=subprocess.PIPE, encoding='utf-8')

  def __send(self, cmd):
    self.proc.stdin.write(cmd)
    self.proc.stdin.flush()

  def pause(self):
    self.__send('p\n')

  def resume(self):
    self.__send('r\n')

  def sync(self):
    self.__send('s\n')

  def shutdown(self):
    self.proc.kill()
