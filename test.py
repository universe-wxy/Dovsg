from xarm.wrapper import XArmAPI

arm = XArmAPI('192.168.1.222')
arm.connect()
print("连接状态:", arm.get_state())
print("版本信息:", arm.version)
print("当前位置:", arm.get_position())
arm.disconnect()
