import imaplib
import email
import datetime

def extract_body(payload):
    if isinstance(payload,str):
        return payload
    else:
        return '\n'.join([extract_body(part.get_payload()) for part in payload])

def retrieve_email(user, password):
    conn = imaplib.IMAP4_SSL("imap.gmail.com", 993)
    conn.login(user, password)
    conn.select()
    typ, data = conn.search(None, 'FROM', '"send mail ONLY to cs"', 'UNSEEN')
    msg_list = []
    try:
        for num in data[0].split():
            typ, msg_data = conn.fetch(num, '(RFC822)')
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    date_time_obj = datetime.datetime.strptime(msg['date'], '%a, %d %b %Y %H:%M:%S %z')
                    payload=msg.get_payload()
                    body=extract_body(payload)
                    msg_list.append({
                        'date': date_time_obj,
                        'body': body
                    })
                    
            typ, response = conn.store(num, '+FLAGS', r'(\Seen)')
    finally:
        try:
            conn.close()
        except:
            pass
        conn.logout()

    return msg_list