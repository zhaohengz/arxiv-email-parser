import imaplib
import email
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--user', type=str, required=True)
parser.add_argument('--password', type=str, required=True)
args = parser.parse_args()

def extract_body(payload):
    if isinstance(payload,str):
        return payload
    else:
        return '\n'.join([extract_body(part.get_payload()) for part in payload])

conn = imaplib.IMAP4_SSL("imap.gmail.com", 993)
conn.login(args.user, args.password)
conn.select()
typ, data = conn.search(None, 'FROM', '"send mail ONLY to cs"', 'UNSEEN')
try:
    for num in data[0].split():
        typ, msg_data = conn.fetch(num, '(RFC822)')
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                subject=msg['subject']                   
                payload=msg.get_payload()
                body=extract_body(payload)
                with open('test.eml', 'w') as fp:
                    fp.write(body)
        typ, response = conn.store(num, '+FLAGS', r'(\Seen)')
finally:
    try:
        conn.close()
    except:
        pass
    conn.logout()