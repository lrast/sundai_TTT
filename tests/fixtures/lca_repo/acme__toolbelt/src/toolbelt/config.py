def parse_config(path):
    data = json.loads(open(path).read())
    return data.get('settings')
