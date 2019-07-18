import re 
import io
import argparse
import gmail
import datetime
import markdown
import os

parser = argparse.ArgumentParser()
parser.add_argument('--user', type=str, required=True)
parser.add_argument('--password', type=str, required=True)
args = parser.parse_args()
  
def find_url(string): 
    # findall() has been used  
    # with valid conditions for urls in string 
    url = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\)]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string) 
    return url 

def extract_info(raw_info):
    # Extract raw info into dict
    dict_info = {}
    raw_lines = raw_info.split('\n')

    idx = 0
    content = ''
    while idx < len(raw_lines):
        pos = raw_lines[idx].find(':')
        if pos >= 0:
            key = raw_lines[idx][:pos]
            content += raw_lines[idx][pos+1:].strip() + ' '
            idx += 1
            if idx >= len(raw_lines):
                break
            pos = raw_lines[idx].find(':')
            while pos < 0 and idx < len(raw_lines):
                content += raw_lines[idx].strip() + ' '
                idx += 1
                if (idx >= len(raw_lines)):
                    break
                pos = raw_lines[idx].find(':')
            dict_info[key] = content
            content = ''
        else:
            idx += 1
    return dict_info
    
def write_to_md(info_list, abstract_list, file_name):

    # Write paper information/abstract to md file
    with open(file_name, 'w') as fp:
        for info, abstract in zip(info_list, abstract_list):    
            fp.write('# [{}]({})\n'.format(info['Title'], info['url']) + '\n')
            fp.write('### Authors: ' + info['Authors'] + '\n')
            fp.write('### Categories: ' + info['Categories'] + '\n')
            if 'Comments' in info:
                fp.write('### Comments: ' + info['Comments'] + '\n')
            fp.write('---\n')
            fp.write(abstract)
            fp.write('\n')

def write_to_html(info_list, abstract_list, file_name):
        
    # Render the markdown string to a html file
    md_str = ''
    for info, abstract in zip(info_list, abstract_list):    
        md_str += '# [{}]({})\n'.format(info['Title'], info['url']) + '\n'
        md_str += '### Authors: ' + info['Authors'] + '\n'
        md_str +='### Categories: ' + info['Categories'] + '\n'
        if 'Comments' in info:
            md_str+= '### Comments: ' + info['Comments'] + '\n'
        md_str +='---\n'
        md_str += abstract
        md_str += '\n'
    
    with open(file_name, 'w') as fp:
        fp.write(markdown.markdown(md_str))

def process_email(msg):

    #Process msg
    buf = io.StringIO(msg['body'])
    raw_lines = buf.readlines()
    idx = 0

    while not '\\\\' in raw_lines[idx]:
        idx += 1

    raw_lines = raw_lines[idx+1:]
    info = []
    abstract = []

    content = ''
    occurance = 0
    for line in raw_lines: 
        if 'replaced with revised version' in line:
            break
        if not '\\\\' in line:
            content += line
        else:
            if occurance % 3 == 1: 
                abstract.append(content.strip())
                url = find_url(line)
                info[-1]['url'] = url[0]
            elif occurance % 3 == 0:
                info.append(extract_info(content))
            content = ''
            occurance += 1

    if not os.path.exists('output/md'):
        os.makedirs('output/md')
    
    if not os.path.exists('output/html'):
        os.makedirs('output/html')
    
    write_to_md(info, abstract, msg['date'].strftime('output/md/%d-%m-%y.md'))
    write_to_html(info, abstract, msg['date'].strftime('output/html/%d-%m-%y.html'))

 
msg_list = gmail.retrieve_email(args.user, args.password)
for msg in msg_list:
    process_email(msg)

   