import importlib, inspect, sys
sys.path.insert(0, '.')
modules = [
    'tests.test_m1',
    'tests.test_m2',
    'tests.test_m3',
    'tests.test_m4',
]
failed = []
for mod_name in modules:
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        print(f'ERROR importing {mod_name}: {e}')
        failed.append((mod_name, 'import', e))
        continue
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith('test_'):
            try:
                fn()
                print(f'OK: {mod_name}.{name}')
            except AssertionError as e:
                print(f'FAIL: {mod_name}.{name} -> {e}')
                failed.append((mod_name + '.' + name, 'assert', e))
            except Exception as e:
                print(f'ERROR: {mod_name}.{name} -> {e}')
                failed.append((mod_name + '.' + name, 'error', e))

print('\nSummary:')
print(f'  Total tests run: TBD')
print(f'  Failures/Errors: {len(failed)}')
if failed:
    for it in failed[:10]:
        print(' -', it)
    sys.exit(2)
else:
    print('All tests passed (by lightweight runner)')
    sys.exit(0)
