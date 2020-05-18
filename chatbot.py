import itchat,time,sys

def output_info(msg):
    print('[info]%s' % msg)

def open_QR():
    for get_count in range(10):
        output_info("getting uuid")
        uuid=itchat.get_QRuuid()
        while uuid is None: uuid=itchat.get_QRuuid(); time.sleep(1)
        output_info("getting QR CODE")
        if itchat.get_QR(uuid): break
        elif get_count >= 9:
            output_info("Failed to get QR Code ,restart the program")
            sys.exit()
    output_info("Please scan the QR Code")
    return uuid

uuid = open_QR()
waitForConfirm = False
while 1:
    status = itchat.check_login(uuid)
    if status == '200':
        break
    elif status == '201':
        if waitForConfirm:
            output_info('please press confirm')
            waitForConfirm = True
    elif status == '408':
        output_info('reload the QR Code')
        uuid = open_QR()
        waitForConfirm = False

userInfo = itchat.web_init()
itchat.show_mobile_login()
itchat.get_contact()
output_info('login successfully as %s' % userInfo['无馅儿包子'])
itchat.start_receiving()

@itchat.msg_register
def simple_reply(msg):
    if msg['type'] == 'TEXT':
        return 'I received : %s ' % msg['Content']
itchat.run()


