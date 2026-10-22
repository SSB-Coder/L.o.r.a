"""Inventory every first-party Python module without importing hardware backends."""
import ast
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
records=[]
for path in sorted((root/'src').rglob('*.py')):
    source=path.read_text(encoding='utf-8-sig')
    tree=ast.parse(source, filename=str(path))
    functions=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    suppressed=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.ExceptHandler) and all(isinstance(x,ast.Pass) for x in n.body)]
    records.append({'file':path.relative_to(root).as_posix(),'lines':len(source.splitlines()),
        'functions':len(functions),'longest_function':max([n.end_lineno-n.lineno+1 for n in functions] or [0]),
        'silent_exception_handlers':suppressed})
report={'modules':len(records),'source_lines':sum(r['lines'] for r in records),
        'functions':sum(r['functions'] for r in records),
        'silent_exception_handlers':sum(len(r['silent_exception_handlers']) for r in records), 'files':records}
out=root/'docs/validation/source-inventory.json'
out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print({k:v for k,v in report.items() if k!='files'})
