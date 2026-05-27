import re

with open('crates/xiang-core/src/hanzi_table.rs', 'r', encoding='utf-8') as f:
    content = f.read()

sq = "'"
content = re.sub(
    r'HanziEntry::([pic])\("([^"]{1,2})"',
    lambda m: f'HanziEntry::{m.group(1)}({sq}{m.group(2)}{sq}',
    content
)

with open('crates/xiang-core/src/hanzi_table.rs', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed quotes in hanzi_table.rs')
