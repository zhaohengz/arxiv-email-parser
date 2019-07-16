
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
        else:
            idx += 1
    return dict_info
    
def write_to_md(info_list, abstract_list, file_name):

    # Write paper information/abstract to md file
    with open(file_name, 'w') as fp:
        idx = 0
        for info, abstract in zip(info_list, abstract_list):
            fp.write('# ' + info['Title'] + '\n')
            fp.write(abstract)
            fp.write('\n')


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
occurance = 0
for line in raw_lines:
    if 'replaced with revised version' in line:
        break
    if not '\\\\' in line:
        content += line
    else:
        if occurance % 3 == 1: 
            abstract.append(content.strip())
        elif occurance % 3 == 0:
            info.append(extract_info(content))
        content = ''
        occurance += 1

write_to_md(info, abstract, 'output.md')

    