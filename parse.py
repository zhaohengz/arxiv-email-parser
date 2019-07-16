
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
            pos = raw_lines[idx].find(':')
            while pos < 0 and idx < len(raw_lines):
                content += raw_lines[idx].strip() + ' '
                idx += 1
                if (idx >= len(raw_lines)):
                    break
                pos = raw_lines[idx].find(':')
            dict_info[key] = content
            content = ''
    
    print (dict_info)


with open('test.eml') as fp:
    raw_lines = fp.readlines()

idx = 0

while not '\\\\' in raw_lines[idx]:
    idx += 1

raw_lines = raw_lines[idx+1:]
info = []
abstract = []
is_abstract = False

content = ''
for line in raw_lines:
    if not '\\\\' in line:
        content += line
    else:
        if is_abstract: 
            abstract.append(content)
        else:
            info.append(content)
        content = ''
        is_abstract = not is_abstract

extract_info(info[0])
    