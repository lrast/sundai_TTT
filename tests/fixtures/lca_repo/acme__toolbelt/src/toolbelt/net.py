def retry_request(url, times):
    resp = None
    for _ in range(times):
        resp = fetch(url)
    return resp
