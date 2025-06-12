import json
import requests


def submit(results, groupname, url="http://tatooine.disi.unitn.it:3001/retrieval/"):
    res = {}
    res["groupname"] = groupname
    res["images"] = results
    res = json.dumps(res)
    response = requests.post(url, res)
    # print(res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
