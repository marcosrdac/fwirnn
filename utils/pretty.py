def pprint(d, indent=0, indenter='  ', arrow='- ', dontprint=()):
    for key, value in d.items():
        if key not in dontprint:
            if isinstance(value, dict):
                print(f'{indenter * indent}{arrow}{key}')
                pprint(value, indent + 1, indenter)
            else:
                print(f'{indenter * indent}{arrow}{key}', end=': ')
                print(repr(value))
