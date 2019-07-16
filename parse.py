
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

print (info[0])
print (abstract[0])
    